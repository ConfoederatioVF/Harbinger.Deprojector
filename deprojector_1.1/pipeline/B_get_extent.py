"""
Pipeline Step B: Extent Finder
Locates the spatial extent and correspondence mapping of a source map inside a reference map.
"""

import os
import sys
import types
import warnings
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree

# Core Dataclasses
from core.dataclass import ExtentResult, NormalizedPair

# Pipeline Step A Utilities
from pipeline.A_coastline_masking import (
    _coast_dt_to_uint8,
    _float_to_uint8,
    _signed_dt_to_uint8,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 1. TORCH & DENSE MATCHER INITIALIZATION & SHIMS
# ═══════════════════════════════════════════════════════════════════

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import kornia.feature as KF
    HAS_LOFTR = True
except ImportError:
    HAS_LOFTR = False

# RoMa Bridge Setup to redirect CUDA kernels to native PyTorch
if HAS_TORCH and "local_corr" not in sys.modules:
    try:
        from romatch.utils import local_correlation as _lc

        shim = types.ModuleType("local_corr")

        def local_corr_bridge(feat0, feat1, warp, mode="bilinear", normalized_coords=True):
            B, C, H, W = feat0.shape
            K = warp.shape[2]
            res, _ = _lc.shitty_native_torch_local_corr(
                feature0=feat0.permute(0, 2, 1).reshape(B, C, int(np.sqrt(H * W)), int(np.sqrt(H * W))),
                feature1=feat1.permute(0, 3, 1, 2),
                warp=warp[..., 0, :].reshape(B, int(np.sqrt(H * W)), int(np.sqrt(H * W)), 2),
                local_window=None,
                B=B, K=K, c=C, r=None, h=int(np.sqrt(H * W)), w=int(np.sqrt(H * W)),
                device=feat0.device,
                sample_mode=mode,
            )
            return res

        shim.local_corr = local_corr_bridge
        sys.modules["local_corr"] = shim
        _lc.use_custom_corr = False
        print("  ✓ RoMa Bridge Active: Redirecting CUDA kernels to native PyTorch")
    except Exception as e:
        mock = types.ModuleType("local_corr")
        mock.local_corr = lambda *args, **kwargs: torch.zeros(1)
        sys.modules["local_corr"] = mock
        print(f"  ! RoMa Bridge failed to initialize ({e}), using dummy mock")

HAS_ROMA_OUTDOOR = False
try:
    from romatch import roma_outdoor
    HAS_ROMA_OUTDOOR = True
except ImportError:
    pass

HAS_ROMA_TINY = False
try:
    try:
        from romatch import roma_tiny
    except ImportError:
        from romatch.models.model_zoo import roma_tiny
    HAS_ROMA_TINY = True
except ImportError:
    pass

HAS_ROMA = HAS_ROMA_OUTDOOR or HAS_ROMA_TINY


# ═══════════════════════════════════════════════════════════════════
# 2. IMAGE PREPROCESSING & MASKING HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_sea_color(img: np.ndarray, sample_size: int = 300) -> np.ndarray:
    small = cv2.resize(img, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3) if len(small.shape) == 3 else small.reshape(-1, 1)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def extract_land_mask(
    img: np.ndarray, threshold: int = 35, min_frac: float = 0.0001
) -> np.ndarray:
    sea = get_sea_color(img).astype(np.float64)
    if len(img.shape) == 3:
        diff = np.linalg.norm(img.astype(np.float64) - sea, axis=2)
    else:
        diff = np.abs(img.astype(np.float64) - float(sea.ravel()[0]))
    mask = (diff > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    min_area = mask.shape[0] * mask.shape[1] * min_frac
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(solid, [c], -1, 255, -1)
    return solid


def make_distance_transform(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    dt_sea = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    dt_land = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    signed = dt_land - dt_sea
    mx = max(np.abs(signed).max(), 1.0)
    return ((signed / mx) * 127 + 128).clip(0, 255).astype(np.uint8)


def make_edge_distance(mask: np.ndarray, max_dist: float = 35.0) -> np.ndarray:
    edges = cv2.Canny(mask, 50, 150)
    inv = cv2.bitwise_not(edges)
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    clamped = np.clip(dt, 0, max_dist)
    return (255 - (clamped / max_dist * 255)).astype(np.uint8)


def resize_to(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(max_dim / max(h, w), 1.0)
    nh, nw = max(int(h * s), 32), max(int(w * s), 32)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════════════
# 3. POLYNOMIAL WARP MODEL
# ═══════════════════════════════════════════════════════════════════

def _poly_design_matrix(pts: np.ndarray, degree: int) -> np.ndarray:
    cols = []
    for total in range(degree + 1):
        for dy in range(total + 1):
            dx = total - dy
            cols.append(pts[:, 0] ** dx * pts[:, 1] ** dy)
    return np.column_stack(cols)


def _n_poly_coeffs(degree: int) -> int:
    return (degree + 1) * (degree + 2) // 2


def fit_polynomial_warp(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    degree: int = 2,
    weights: Optional[np.ndarray] = None,
) -> dict:
    src_mean = src_pts.mean(axis=0)
    src_std = src_pts.std(axis=0)
    src_std[src_std < 1e-10] = 1.0
    src_norm = (src_pts - src_mean) / src_std

    A = _poly_design_matrix(src_norm, degree)

    if weights is not None:
        W = np.sqrt(weights)
        A = A * W[:, None]
        rx = ref_pts[:, 0] * W
        ry = ref_pts[:, 1] * W
    else:
        rx = ref_pts[:, 0]
        ry = ref_pts[:, 1]

    coeffs_x, _, _, _ = np.linalg.lstsq(A, rx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, ry, rcond=None)

    return {
        "coeffs_x": coeffs_x,
        "coeffs_y": coeffs_y,
        "degree": degree,
        "src_mean": src_mean,
        "src_std": src_std,
    }


def apply_polynomial_warp(pts: np.ndarray, warp: dict) -> np.ndarray:
    pts_norm = (pts - warp["src_mean"]) / warp["src_std"]
    A = _poly_design_matrix(pts_norm, warp["degree"])
    ref_x = A @ warp["coeffs_x"]
    ref_y = A @ warp["coeffs_y"]
    return np.column_stack([ref_x, ref_y])


def polynomial_warp_error(
    src_pts: np.ndarray, ref_pts: np.ndarray, warp: dict
) -> np.ndarray:
    predicted = apply_polynomial_warp(src_pts, warp)
    return np.linalg.norm(predicted - ref_pts, axis=1)


# ═══════════════════════════════════════════════════════════════════
# 4. CENTER-WEIGHTED PROGRESSIVE RANSAC
# ═══════════════════════════════════════════════════════════════════

def _center_weights(src_pts: np.ndarray, src_shape: Tuple[int, ...]) -> np.ndarray:
    center = np.array([src_shape[1] / 2.0, src_shape[0] / 2.0])
    max_r = np.linalg.norm(center)
    dists = np.linalg.norm(src_pts - center, axis=1)
    return np.exp(-2.0 * (dists / max_r) ** 2)


def center_weighted_ransac(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    confs: np.ndarray,
    src_shape: Tuple[int, ...],
    ref_shape: Tuple[int, ...],
    degree: int = 2,
    n_iters: int = 5000,
    inlier_frac: float = 0.025,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict, np.ndarray]:
    n = len(src_pts)
    min_samples = _n_poly_coeffs(degree) + 2

    if n < min_samples:
        if verbose:
            print(f"  Only {n} points, need {min_samples}. Fitting on all.")
        warp = fit_polynomial_warp(src_pts, ref_pts, degree)
        return np.ones(n, dtype=bool), warp, np.zeros(n)

    diag = np.sqrt(ref_shape[0] ** 2 + ref_shape[1] ** 2)
    thresh = diag * inlier_frac

    cw = _center_weights(src_pts, src_shape)
    probs = confs * cw
    probs /= probs.sum()

    best_score = -1
    best_mask = np.zeros(n, dtype=bool)
    best_warp = None

    for it in range(n_iters):
        try:
            idx = np.random.choice(n, size=min_samples, replace=False, p=probs)
        except ValueError:
            idx = np.random.choice(n, size=min_samples, replace=False)

        s_sample = src_pts[idx]
        r_sample = ref_pts[idx]

        src_span = np.ptp(s_sample, axis=0)
        if src_span[0] < 20 or src_span[1] < 20:
            continue

        try:
            warp = fit_polynomial_warp(s_sample, r_sample, degree)
            errors = polynomial_warp_error(src_pts, ref_pts, warp)
        except Exception:
            continue

        mask = errors < thresh
        score = (mask * confs * cw).sum()

        if score > best_score:
            best_score = score
            best_mask = mask.copy()
            best_warp = warp

            if mask.sum() > 0.85 * n:
                break

    if best_mask.sum() >= min_samples:
        inlier_cw = _center_weights(src_pts[best_mask], src_shape)
        best_warp = fit_polynomial_warp(
            src_pts[best_mask],
            ref_pts[best_mask],
            degree,
            weights=inlier_cw * confs[best_mask],
        )
        errors = polynomial_warp_error(src_pts, ref_pts, best_warp)
        best_mask = errors < thresh

        if best_mask.sum() >= min_samples:
            inlier_cw = _center_weights(src_pts[best_mask], src_shape)
            best_warp = fit_polynomial_warp(
                src_pts[best_mask],
                ref_pts[best_mask],
                degree,
                weights=inlier_cw * confs[best_mask],
            )

    errors = polynomial_warp_error(src_pts, ref_pts, best_warp)

    if verbose:
        print(f"  RANSAC: {best_mask.sum()}/{n} inliers, thresh={thresh:.1f}px, degree={degree}")

    return best_mask, best_warp, errors


def progressive_ransac(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    confs: np.ndarray,
    src_shape: Tuple[int, ...],
    ref_shape: Tuple[int, ...],
    verbose: bool = False,
) -> Tuple[np.ndarray, dict, np.ndarray]:
    n = len(src_pts)
    center = np.array([src_shape[1] / 2, src_shape[0] / 2])
    dists = np.linalg.norm(src_pts - center, axis=1)
    max_dist = dists.max()

    phases = [
        (0.35, 1, 0.030, 2000),
        (0.65, 2, 0.025, 3000),
        (1.00, 2, 0.025, 3000),
        (1.00, 3, 0.020, 3000),
    ]

    current_mask = np.zeros(n, dtype=bool)
    current_warp = None

    for phase_i, (dist_frac, degree, inlier_frac, iters) in enumerate(phases):
        min_needed = _n_poly_coeffs(degree) + 2
        dist_thresh = max_dist * dist_frac
        spatial_mask = dists <= dist_thresh

        candidate_mask = spatial_mask | current_mask
        candidate_idx = np.where(candidate_mask)[0]

        if len(candidate_idx) < min_needed:
            if verbose:
                print(f"  Phase {phase_i}: skip ({len(candidate_idx)} < {min_needed})")
            continue

        mask_local, warp, errors = center_weighted_ransac(
            src_pts[candidate_idx],
            ref_pts[candidate_idx],
            confs[candidate_idx],
            src_shape,
            ref_shape,
            degree=degree,
            n_iters=iters,
            inlier_frac=inlier_frac,
            verbose=False,
        )

        n_inliers = mask_local.sum()

        if verbose:
            print(f"  Phase {phase_i}: dist≤{dist_frac:.0%}, deg={degree}, {n_inliers}/{len(candidate_idx)} inliers")

        if n_inliers >= min_needed:
            global_inliers = candidate_idx[mask_local]
            current_mask[:] = False
            current_mask[global_inliers] = True
            current_warp = warp

    if current_warp is None:
        if verbose:
            print("  Progressive RANSAC failed, falling back to full fit")
        current_mask, current_warp, errors = center_weighted_ransac(
            src_pts,
            ref_pts,
            confs,
            src_shape,
            ref_shape,
            degree=2,
            n_iters=5000,
            inlier_frac=0.03,
            verbose=verbose,
        )

    errors = polynomial_warp_error(src_pts, ref_pts, current_warp)
    return current_mask, current_warp, errors


# ═══════════════════════════════════════════════════════════════════
# 5. INITIAL ANCHOR VIA TEMPLATE MATCHING
# ═══════════════════════════════════════════════════════════════════

def find_center_anchor(
    ref_dt: np.ndarray,
    src_dt: np.ndarray,
    center_frac: float = 0.25,
    n_scales: int = 30,
    scale_range: Tuple[float, float] = (0.2, 3.0),
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], float, float]:
    sh, sw = src_dt.shape[:2]
    rh, rw = ref_dt.shape[:2]

    cy, cx = sh // 2, sw // 2
    ch = int(sh * center_frac)
    cw = int(sw * center_frac)
    template = src_dt[
        cy - ch // 2 : cy + ch // 2,
        cx - cw // 2 : cx + cw // 2,
    ]

    if template.shape[0] < 16 or template.shape[1] < 16:
        return None, 1.0, 0.0

    best_val = -1.0
    best_pos = None
    best_scale = 1.0

    scales = np.linspace(scale_range[0], scale_range[1], n_scales)

    for scale in scales:
        th = int(template.shape[0] * scale)
        tw = int(template.shape[1] * scale)
        if th >= rh - 2 or tw >= rw - 2 or th < 16 or tw < 16:
            continue

        t_scaled = cv2.resize(
            template,
            (tw, th),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

        result = cv2.matchTemplate(ref_dt, t_scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_pos = np.array([max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0])
            best_scale = scale

    if verbose and best_pos is not None:
        print(f"  Anchor: ref=({best_pos[0]:.0f}, {best_pos[1]:.0f}), scale={best_scale:.2f}, NCC={best_val:.3f}")

    return best_pos, best_scale, best_val


# ═══════════════════════════════════════════════════════════════════
# 6. MATCHING FRONT-ENDS
# ═══════════════════════════════════════════════════════════════════

def _to_loftr_tensor(img: np.ndarray, device) -> "torch.Tensor":
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    t = torch.from_numpy(gray).float() / 255.0
    return t[None, None].to(device)


def match_loftr(
    ref_repr: np.ndarray,
    src_repr: np.ndarray,
    ref_full_shape: Tuple[int, ...],
    src_full_shape: Tuple[int, ...],
    confidence_thresh: float = 0.2,
    device=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_LOFTR:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
    ref_t = _to_loftr_tensor(ref_repr, device)
    src_t = _to_loftr_tensor(src_repr, device)
    with torch.inference_mode():
        out = matcher({"image0": ref_t, "image1": src_t})
    kp_r = out["keypoints0"].cpu().numpy()
    kp_s = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    keep = conf >= confidence_thresh
    kp_r, kp_s, conf = kp_r[keep], kp_s[keep], conf[keep]
    if len(kp_r) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    rh, rw = ref_repr.shape[:2]
    sh, sw = src_repr.shape[:2]
    ref_pts = kp_r.astype(np.float64)
    src_pts = kp_s.astype(np.float64)
    ref_pts[:, 0] *= ref_full_shape[1] / rw
    ref_pts[:, 1] *= ref_full_shape[0] / rh
    src_pts[:, 0] *= src_full_shape[1] / sw
    src_pts[:, 1] *= src_full_shape[0] / sh
    return ref_pts, src_pts, conf


def match_roma(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    ref_full_shape: Tuple[int, ...],
    src_full_shape: Tuple[int, ...],
    n_matches: int = 2000,
    device=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_ROMA:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if HAS_ROMA_TINY:
        try:
            model = roma_tiny(device=device)
        except Exception as e:
            print(f"  RoMa Tiny init failed: {e}")

    if model is None and HAS_ROMA_OUTDOOR:
        try:
            model = roma_outdoor(device=device)
        except Exception as e:
            print(f"  RoMa Outdoor failed (likely missing 'local_corr'): {e}")

    if model is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)

    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) if len(ref_img.shape) == 3 else cv2.cvtColor(ref_img, cv2.COLOR_GRAY2RGB)
    src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB) if len(src_img.shape) == 3 else cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)

    ref_pil = Image.fromarray(ref_rgb)
    src_pil = Image.fromarray(src_rgb)

    with torch.inference_mode():
        warp, certainty = model.match(ref_pil, src_pil, device=device)
        matches, conf = model.sample(warp, certainty, num=n_matches)

    matches_np = matches.cpu().numpy().astype(np.float64)
    ref_pts_norm = matches_np[:, :2]
    src_pts_norm = matches_np[:, 2:]
    confs = conf.cpu().numpy().astype(np.float64)

    ref_pts = np.zeros_like(ref_pts_norm)
    ref_pts[:, 0] = (ref_pts_norm[:, 0] + 1) / 2 * ref_full_shape[1]
    ref_pts[:, 1] = (ref_pts_norm[:, 1] + 1) / 2 * ref_full_shape[0]

    src_pts = np.zeros_like(src_pts_norm)
    src_pts[:, 0] = (src_pts_norm[:, 0] + 1) / 2 * src_full_shape[1]
    src_pts[:, 1] = (src_pts_norm[:, 1] + 1) / 2 * src_full_shape[0]

    return ref_pts, src_pts, confs


def match_sift(
    ref_repr: np.ndarray,
    src_repr: np.ndarray,
    ref_full_shape: Tuple[int, ...],
    src_full_shape: Tuple[int, ...],
    n_features: int = 20000,
    ratio_thresh: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_r, des_r = sift.detectAndCompute(ref_repr, None)
    kp_s, des_s = sift.detectAndCompute(src_repr, None)
    if des_r is None or des_s is None or len(des_r) < 2 or len(des_s) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des_s, des_r, k=2)
    good = [m for m, n in raw if m.distance < ratio_thresh * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    src_pts = np.array([kp_s[m.queryIdx].pt for m in good], dtype=np.float64)
    ref_pts = np.array([kp_r[m.trainIdx].pt for m in good], dtype=np.float64)
    confs = np.array([max(0.01, 1.0 - m.distance / 500.0) for m in good], dtype=np.float64)
    rh, rw = ref_repr.shape[:2]
    sh, sw = src_repr.shape[:2]
    ref_pts[:, 0] *= ref_full_shape[1] / rw
    ref_pts[:, 1] *= ref_full_shape[0] / rh
    src_pts[:, 0] *= src_full_shape[1] / sw
    src_pts[:, 1] *= src_full_shape[0] / sh
    return ref_pts, src_pts, confs


# ═══════════════════════════════════════════════════════════════════
# 7. DEDUPLICATION & ANCHOR FILTERING
# ═══════════════════════════════════════════════════════════════════

def deduplicate(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    confs: np.ndarray,
    src_shape: Tuple[int, ...],
    ref_shape: Tuple[int, ...],
    r: float = 0.008,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(src_pts) < 2:
        return src_pts, ref_pts, confs
    combined = np.column_stack(
        [
            src_pts / [src_shape[1], src_shape[0]],
            ref_pts / [ref_shape[1], ref_shape[0]],
        ]
    )
    tree = cKDTree(combined)
    pairs = tree.query_pairs(r=r)
    remove = set()
    for i, j in pairs:
        if confs[i] >= confs[j]:
            remove.add(j)
        else:
            remove.add(i)
    keep = np.array([i for i in range(len(src_pts)) if i not in remove])
    if len(keep) == 0:
        return src_pts[:1], ref_pts[:1], confs[:1]
    return src_pts[keep], ref_pts[keep], confs[keep]


def anchor_filter(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    confs: np.ndarray,
    src_shape: Tuple[int, ...],
    anchor_ref: np.ndarray,
    anchor_scale: float,
    tolerance: float = 2.5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_center = np.array([src_shape[1] / 2.0, src_shape[0] / 2.0])
    expected_radius = np.linalg.norm([src_shape[1] / 2.0, src_shape[0] / 2.0]) * anchor_scale

    offsets = (src_pts - src_center) * anchor_scale
    predicted_ref = anchor_ref + offsets

    errors = np.linalg.norm(ref_pts - predicted_ref, axis=1)
    thresh = expected_radius * tolerance

    keep = errors < thresh
    if verbose:
        print(f"  Anchor filter: {keep.sum()}/{len(keep)} kept (thresh={thresh:.0f}px)")

    if keep.sum() < 4:
        return src_pts, ref_pts, confs

    return src_pts[keep], ref_pts[keep], confs[keep]


# ═══════════════════════════════════════════════════════════════════
# 8. BOUNDARY PROJECTION
# ═══════════════════════════════════════════════════════════════════

def project_boundary(
    src_shape: Tuple[int, ...],
    warp: dict,
    n_per_edge: int = 150,
) -> np.ndarray:
    sh, sw = src_shape[:2]
    n = n_per_edge

    top = np.column_stack([np.linspace(0, sw, n, endpoint=False), np.zeros(n)])
    right = np.column_stack([np.full(n, sw), np.linspace(0, sh, n, endpoint=False)])
    bottom = np.column_stack([np.linspace(sw, 0, n, endpoint=False), np.full(n, sh)])
    left = np.column_stack([np.zeros(n), np.linspace(sh, 0, n, endpoint=False)])

    border = np.vstack([top, right, bottom, left])
    return apply_polynomial_warp(border, warp)


def project_boundary_with_sanity(
    src_shape: Tuple[int, ...],
    ref_shape: Tuple[int, ...],
    warp: dict,
    n_per_edge: int = 150,
    margin_frac: float = 0.5,
) -> np.ndarray:
    polygon = project_boundary(src_shape, warp, n_per_edge)

    rh, rw = ref_shape[:2]
    margin_x = rw * margin_frac
    margin_y = rh * margin_frac

    polygon[:, 0] = np.clip(polygon[:, 0], -margin_x, rw + margin_x)
    polygon[:, 1] = np.clip(polygon[:, 1], -margin_y, rh + margin_y)

    return polygon


# ═══════════════════════════════════════════════════════════════════
# 9. MAIN PIPELINE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def find_extent(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    precomputed_pair: Optional[NormalizedPair] = None,
    mask_threshold: int = 35,
    work_max_dim: int = 840,
    loftr_dim: int = 640,
    poly_degree: int = 2,
    ransac_iters: int = 5000,
    ransac_inlier_frac: float = 0.025,
    use_anchor: bool = True,
    verbose: bool = True,
) -> ExtentResult:
    ref_full = ref_img.shape
    src_full = src_img.shape

    if verbose:
        print("═" * 60)
        print("  MAP EXTENT FINDER v5 — Preprocessed Segmentation")
        print("═" * 60)

    # ── Masks & Representations ──────────────────────────────────
    if precomputed_pair is not None:
        pair = precomputed_pair
        if verbose:
            print("\n①② Using precomputed segmentation pair")
            print(f"   ref seg conf: {pair.ref_seg.confidence:.3f}")
            print(f"   src seg conf: {pair.src_seg.confidence:.3f}")

        ref_mask = pair.ref_seg.land_mask
        src_mask = pair.src_seg.land_mask
        ref_mask_w = pair.ref_land_mask
        src_mask_w = pair.src_land_mask

        ref_dt = _signed_dt_to_uint8(pair.ref_dt_signed)
        src_dt = _signed_dt_to_uint8(pair.src_dt_signed)

        ref_blur = cv2.GaussianBlur(ref_mask_w, (0, 0), 4.0)
        src_blur = cv2.GaussianBlur(src_mask_w, (0, 0), 4.0)

        ref_edge = _coast_dt_to_uint8(pair.ref_coastline_dt)
        src_edge = _coast_dt_to_uint8(pair.src_coastline_dt)

        ref_curv_u8 = _float_to_uint8(pair.ref_curvature)
        src_curv_u8 = _float_to_uint8(pair.src_curvature)
    else:
        if verbose:
            print("\n① Extracting land masks (legacy mode) …")
            print(f"   ref: {ref_full[1]}×{ref_full[0]}")
            print(f"   src: {src_full[1]}×{src_full[0]}")

        ref_mask = extract_land_mask(ref_img, threshold=mask_threshold)
        src_mask = extract_land_mask(src_img, threshold=mask_threshold)

        if verbose:
            rl = (ref_mask > 0).mean() * 100
            sl = (src_mask > 0).mean() * 100
            print(f"   ref land: {rl:.1f}%  src land: {sl:.1f}%")
            print("\n② Creating representations …")

        ref_mask_w = resize_to(ref_mask, work_max_dim)
        src_mask_w = resize_to(src_mask, work_max_dim)

        ref_dt = make_distance_transform(ref_mask_w)
        src_dt = make_distance_transform(src_mask_w)

        ref_blur = cv2.GaussianBlur(ref_mask_w, (0, 0), 4.0)
        src_blur = cv2.GaussianBlur(src_mask_w, (0, 0), 4.0)

        ref_edge = make_edge_distance(ref_mask_w)
        src_edge = make_edge_distance(src_mask_w)

        ref_curv_u8 = None
        src_curv_u8 = None

    # ── Template Anchor ──────────────────────────────────────────
    anchor_ref = None
    anchor_scale = 1.0

    if use_anchor:
        if verbose:
            print("\n③ Template matching for center anchor …")

        anchor_ref_work, anchor_scale_work, anchor_ncc = find_center_anchor(
            ref_dt,
            src_dt,
            center_frac=0.25,
            n_scales=40,
            scale_range=(0.1, 4.0),
            verbose=verbose,
        )

        if anchor_ref_work is not None and anchor_ncc > 0.3:
            ref_sx = ref_full[1] / ref_dt.shape[1]
            ref_sy = ref_full[0] / ref_dt.shape[0]
            anchor_ref = np.array([anchor_ref_work[0] * ref_sx, anchor_ref_work[1] * ref_sy])
            src_sx = src_full[1] / src_dt.shape[1]
            anchor_scale = anchor_scale_work * ref_sx / src_sx

            if verbose:
                print(
                    f"  Full-res anchor: ({anchor_ref[0]:.0f}, {anchor_ref[1]:.0f}), "
                    f"scale={anchor_scale:.3f}, NCC={anchor_ncc:.3f}"
                )
        else:
            if verbose:
                print(f"  Anchor weak (NCC={anchor_ncc:.3f}), skipping")
            anchor_ref = None

    # ── Matching Phase ───────────────────────────────────────────
    all_ref: List[np.ndarray] = []
    all_src: List[np.ndarray] = []
    all_conf: List[np.ndarray] = []
    methods: List[str] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None

    # RoMa
    if HAS_ROMA:
        if verbose:
            print("\n④a RoMa dense matching …")
        try:
            r, s, c = match_roma(ref_mask_w, src_mask_w, ref_full, src_full, n_matches=2000)
            if len(r) > 0:
                if verbose:
                    print(f"   RoMa: {len(r)} matches")
                all_ref.append(r)
                all_src.append(s)
                all_conf.append(c)
        except Exception as e:
            if verbose:
                print(f"   RoMa execution failed: {e}")
    elif verbose:
        print("    RoMa could not be found.")

    # LoFTR
    if HAS_LOFTR:
        if verbose:
            print("\n④b LoFTR matching …")

        repr_pairs = {
            "dt": (ref_dt, src_dt),
            "blur": (ref_blur, src_blur),
            "edge": (ref_edge, src_edge),
            "mask": (ref_mask_w, src_mask_w),
        }

        if ref_curv_u8 is not None and src_curv_u8 is not None:
            repr_pairs["curvature"] = (ref_curv_u8, src_curv_u8)

        for name, (rr, sr) in repr_pairs.items():
            rr_l = resize_to(rr, loftr_dim)
            sr_l = resize_to(sr, loftr_dim)
            for ct in [0.15, 0.3, 0.5]:
                try:
                    r, s, c = match_loftr(
                        rr_l, sr_l, ref_full, src_full, confidence_thresh=ct, device=device
                    )
                    if verbose and len(r) > 0:
                        print(f"   LoFTR [{name}, ≥{ct}]: {len(r)}")
                    if len(r) > 0:
                        all_ref.append(r)
                        all_src.append(s)
                        all_conf.append(c)
                        methods.append(f"loftr_{name}_{ct}")
                except Exception as e:
                    if verbose:
                        print(f"   LoFTR [{name}] err: {e}")

    # SIFT
    if verbose:
        print("\n④c SIFT matching …")

    sift_pairs = {
        "dt": (ref_dt, src_dt),
        "blur": (ref_blur, src_blur),
        "edge": (ref_edge, src_edge),
    }
    if ref_curv_u8 is not None and src_curv_u8 is not None:
        sift_pairs["curvature"] = (ref_curv_u8, src_curv_u8)

    for name, (rr, sr) in sift_pairs.items():
        for ratio in [0.65, 0.75, 0.85]:
            r, s, c = match_sift(rr, sr, ref_full, src_full, ratio_thresh=ratio)
            if len(r) > 0:
                if verbose:
                    print(f"   SIFT [{name}, r={ratio}]: {len(r)}")
                all_ref.append(r)
                all_src.append(s)
                all_conf.append(c)
                methods.append(f"sift_{name}_{ratio}")

    # Combine
    if not all_ref:
        if verbose:
            print("\n✗ No matches from any method.")
        return ExtentResult(
            polygon=None, n_correspondences=0, confidence=0.0, method="none"
        )

    ref_pts = np.vstack(all_ref)
    src_pts = np.vstack(all_src)
    confs = np.concatenate(all_conf)

    if verbose:
        print(f"\n⑤ Combined: {len(ref_pts)} raw matches from {len(methods)} configs")

    # Anchor filter
    if anchor_ref is not None:
        if verbose:
            print("\n⑥ Anchor-guided filtering …")
        src_pts, ref_pts, confs = anchor_filter(
            src_pts, ref_pts, confs, src_full, anchor_ref, anchor_scale, tolerance=2.5, verbose=verbose
        )

    # Dedup
    src_pts, ref_pts, confs = deduplicate(
        src_pts, ref_pts, confs, src_full, ref_full, r=0.008
    )
    if verbose:
        print(f"   After dedup: {len(src_pts)}")

    if len(src_pts) < _n_poly_coeffs(poly_degree) + 2:
        if verbose:
            print("\n✗ Too few correspondences.")
        return ExtentResult(
            polygon=None,
            n_correspondences=len(src_pts),
            confidence=0.0,
            method=",".join(methods),
            src_pts=src_pts,
            ref_pts=ref_pts,
        )

    # RANSAC
    if verbose:
        print("\n⑦ Progressive center-weighted RANSAC …")

    inlier_mask, warp, errors = progressive_ransac(
        src_pts, ref_pts, confs, src_full, ref_full, verbose=verbose
    )

    n_inliers = inlier_mask.sum()
    if verbose:
        print(f"   Final inliers: {n_inliers}/{len(src_pts)}")
        median_err = np.median(errors[inlier_mask]) if n_inliers > 0 else 999
        print(f"   Median inlier error: {median_err:.1f}px")

    # Boundary Projection
    if verbose:
        print("\n⑧ Projecting boundary …")

    polygon = project_boundary_with_sanity(
        src_full, ref_full, warp, n_per_edge=150, margin_frac=0.3
    )

    in_bounds = (
        (polygon[:, 0] >= 0)
        & (polygon[:, 0] < ref_full[1])
        & (polygon[:, 1] >= 0)
        & (polygon[:, 1] < ref_full[0])
    )
    boundary_conf = in_bounds.sum() / len(in_bounds)
    inlier_conf = n_inliers / max(len(src_pts), 1)
    confidence = 0.6 * boundary_conf + 0.4 * inlier_conf

    polygon[:, 0] = np.clip(polygon[:, 0], 0, ref_full[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, ref_full[0] - 1)

    if verbose:
        bbox = [
            polygon[:, 0].min(),
            polygon[:, 1].min(),
            polygon[:, 0].max(),
            polygon[:, 1].max(),
        ]
        print(f"\n{'═' * 60}")
        print(f"  ✓ DONE — degree-{warp['degree']} polynomial")
        print(f"{'═' * 60}")
        print(f"   Inliers: {n_inliers}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}] → [{bbox[2]:.0f}, {bbox[3]:.0f}]")

    return ExtentResult(
        polygon=polygon,
        n_correspondences=n_inliers,
        confidence=confidence,
        method=",".join(methods),
        src_pts=src_pts[inlier_mask],
        ref_pts=ref_pts[inlier_mask],
        inlier_mask=inlier_mask,
        warp_coeffs=warp,
        debug={
            "all_src_pts": src_pts,
            "all_ref_pts": ref_pts,
            "all_confs": confs,
            "all_errors": errors,
            "boundary_conf": boundary_conf,
            "inlier_conf": inlier_conf,
            "anchor_ref": anchor_ref,
            "anchor_scale": anchor_scale,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 10. VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def draw_extent(
    ref_img: np.ndarray,
    result: ExtentResult,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    vis = ref_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if result.polygon is None:
        return vis
    pts = result.polygon.astype(np.int32)
    cv2.polylines(vis, [pts], True, color, thickness, cv2.LINE_AA)
    return vis


def visualize_full(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    result: ExtentResult,
    figsize: Tuple[int, int] = (24, 8),
):
    n_plots = 4 if result.warp_coeffs is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    show_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB) if len(src_img.shape) == 3 else src_img
    axes[0].imshow(show_src, cmap="gray")
    axes[0].set_title(f"Source\n{src_img.shape[1]}×{src_img.shape[0]}")
    axes[0].axis("off")

    vis = draw_extent(ref_img, result)
    show_ref = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) if len(vis.shape) == 3 else vis
    axes[1].imshow(show_ref, cmap="gray")
    axes[1].set_title(
        f"Reference + extent\nconf={result.confidence:.2f}, {result.n_correspondences} inliers"
    )
    axes[1].axis("off")

    if result.ref_pts is not None and len(result.ref_pts) > 0:
        vis2 = ref_img.copy()
        if len(vis2.shape) == 2:
            vis2 = cv2.cvtColor(vis2, cv2.COLOR_GRAY2BGR)
        n_show = min(len(result.ref_pts), 300)
        idx = np.random.choice(len(result.ref_pts), n_show, replace=False)
        for i in idx:
            rx, ry = int(result.ref_pts[i, 0]), int(result.ref_pts[i, 1])
            cv2.circle(vis2, (rx, ry), 4, (0, 0, 255), -1)
        if result.polygon is not None:
            pts = result.polygon.astype(np.int32)
            cv2.polylines(vis2, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        show_v2 = cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB) if len(vis2.shape) == 3 else vis2
        axes[2].imshow(show_v2, cmap="gray")
        axes[2].set_title(f"Inlier points ({n_show} shown)")
    else:
        axes[2].set_title("No correspondences")
    axes[2].axis("off")

    if n_plots == 4 and result.polygon is not None:
        rh, rw = ref_img.shape[:2]
        overlay = ref_img.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        mask_poly = np.zeros((rh, rw), dtype=np.uint8)
        pts = result.polygon.astype(np.int32)
        cv2.fillPoly(mask_poly, [pts], 255)
        color_overlay = np.zeros_like(overlay)
        color_overlay[:, :, 1] = mask_poly
        overlay = cv2.addWeighted(overlay, 0.7, color_overlay, 0.3, 0)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[3].set_title("Extent overlay")
        axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_correspondences(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    result: ExtentResult,
    n_show: int = 60,
    figsize: Tuple[int, int] = (22, 8),
):
    if result.src_pts is None or len(result.src_pts) == 0:
        print("No correspondences to visualize.")
        return

    ref_h, ref_w = ref_img.shape[:2]
    src_h, src_w = src_img.shape[:2]
    scale = ref_h / src_h
    src_resized = cv2.resize(src_img, (int(src_w * scale), ref_h), interpolation=cv2.INTER_AREA)
    src_rw = src_resized.shape[1]

    canvas = np.zeros((ref_h, ref_w + src_rw, 3), dtype=np.uint8)
    if len(ref_img.shape) == 2:
        canvas[:, :ref_w] = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        canvas[:, ref_w:] = cv2.cvtColor(src_resized, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:, :ref_w] = ref_img
        canvas[:, ref_w:] = src_resized

    n = min(n_show, len(result.src_pts))
    idx = np.random.choice(len(result.src_pts), n, replace=False)
    colors = plt.cm.hsv(np.linspace(0, 0.9, n))[:, :3] * 255

    for ci, i in enumerate(idx):
        rx = int(result.ref_pts[i, 0])
        ry = int(result.ref_pts[i, 1])
        sx = int(result.src_pts[i, 0] * scale) + ref_w
        sy = int(result.src_pts[i, 1] * scale)
        color = tuple(int(c) for c in colors[ci])
        cv2.circle(canvas, (rx, ry), 5, color, -1)
        cv2.circle(canvas, (sx, sy), 5, color, -1)
        cv2.line(canvas, (rx, ry), (sx, sy), color, 1, cv2.LINE_AA)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Correspondences ({n} of {len(result.src_pts)} inliers)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_ransac(
    ref_img: np.ndarray,
    result: ExtentResult,
    figsize: Tuple[int, int] = (16, 8),
):
    debug = result.debug
    if "all_ref_pts" not in debug:
        print("No debug data.")
        return

    all_ref = debug["all_ref_pts"]
    mask = result.inlier_mask

    vis = ref_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if mask is not None and len(mask) == len(all_ref):
        for i in range(len(all_ref)):
            x, y = int(all_ref[i, 0]), int(all_ref[i, 1])
            if not mask[i]:
                cv2.circle(vis, (x, y), 3, (0, 0, 200), -1)
        for i in range(len(all_ref)):
            x, y = int(all_ref[i, 0]), int(all_ref[i, 1])
            if mask[i]:
                cv2.circle(vis, (x, y), 4, (0, 220, 0), -1)

    if result.polygon is not None:
        pts = result.polygon.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    n_in = mask.sum() if mask is not None else 0
    n_out = len(all_ref) - n_in
    ax.set_title(f"RANSAC: {n_in} inliers (green) / {n_out} outliers (red)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_warp_grid(
    src_shape: Tuple[int, ...],
    ref_img: np.ndarray,
    warp: dict,
    grid_n: int = 20,
    figsize: Tuple[int, int] = (14, 8),
):
    sh, sw = src_shape[:2]
    vis = ref_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i in range(grid_n + 1):
        y = sh * i / grid_n
        line_src = np.column_stack([np.linspace(0, sw, 200), np.full(200, y)])
        line_ref = apply_polynomial_warp(line_src, warp)
        pts = line_ref.astype(np.int32)
        for j in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[j]), tuple(pts[j + 1]), (0, 200, 200), 1, cv2.LINE_AA)

    for i in range(grid_n + 1):
        x = sw * i / grid_n
        line_src = np.column_stack([np.full(200, x), np.linspace(0, sh, 200)])
        line_ref = apply_polynomial_warp(line_src, warp)
        pts = line_ref.astype(np.int32)
        for j in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[j]), tuple(pts[j + 1]), (0, 200, 200), 1, cv2.LINE_AA)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Warped grid (degree {warp['degree']})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()