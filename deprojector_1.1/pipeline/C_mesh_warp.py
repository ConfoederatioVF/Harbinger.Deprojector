"""
Pipeline Step C: Polygon-Constrained Mesh Warp (v2 — Hard-Bounded)
Optimizes a coarse-to-fine deformable displacement mesh constrained within a polygon boundary.
"""

from typing import Tuple, Optional, Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Custom project imports
from core.dataclass import ExtentResult
from pipeline.B_get_extent import (
    extract_land_mask,
    get_sea_color,
    fit_polynomial_warp,
    apply_polynomial_warp,
)


# ─────────────────────────────────────────────────────────────────────
# 1. GAUSSIAN SMOOTHING UTILITY
# ─────────────────────────────────────────────────────────────────────

def gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.1:
        return tensor

    H, W = tensor.shape[2], tensor.shape[3]
    ks = max(3, int(6 * sigma) | 1)

    max_ks = min(H, W)
    if max_ks < 3:
        return tensor
    if ks > max_ks:
        ks = max_ks if max_ks % 2 == 1 else max_ks - 1
    if ks < 3:
        return tensor

    ax = torch.arange(
        -(ks // 2),
        ks // 2 + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    C = tensor.shape[1]
    kh = kernel_1d.view(1, 1, 1, -1).expand(C, -1, -1, -1)
    kv = kernel_1d.view(1, 1, -1, 1).expand(C, -1, -1, -1)
    pad = ks // 2

    out = F.conv2d(
        F.pad(tensor, [pad, pad, 0, 0], mode="reflect"),
        kh,
        groups=C,
    )
    out = F.conv2d(
        F.pad(out, [0, 0, pad, pad], mode="reflect"),
        kv,
        groups=C,
    )
    return out


# ─────────────────────────────────────────────────────────────────────
# 2. POLYGON UTILITIES
# ─────────────────────────────────────────────────────────────────────

def polygon_tight_bbox(
    polygon: np.ndarray, ref_shape: Tuple[int, int]
) -> dict:
    rh, rw = ref_shape[:2]
    x0 = int(np.floor(polygon[:, 0].min()))
    y0 = int(np.floor(polygon[:, 1].min()))
    x1 = int(np.ceil(polygon[:, 0].max()))
    y1 = int(np.ceil(polygon[:, 1].max()))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(rw - 1, x1)
    y1 = min(rh - 1, y1)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def rasterise_polygon_mask(
    polygon: np.ndarray,
    h: int,
    w: int,
    offset_xy: Optional[Tuple[int, int]] = None,
    scale_xy: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    poly = polygon.copy().astype(np.float64)
    if offset_xy is not None:
        poly[:, 0] -= offset_xy[0]
        poly[:, 1] -= offset_xy[1]
    if scale_xy is not None:
        poly[:, 0] *= scale_xy[0]
        poly[:, 1] *= scale_xy[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def polygon_mask_tensor(
    polygon: np.ndarray,
    h: int,
    w: int,
    offset_xy: Optional[Tuple[int, int]] = None,
    scale_xy: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    mask_np = rasterise_polygon_mask(polygon, h, w, offset_xy, scale_xy)
    t = torch.from_numpy(mask_np).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


# ─────────────────────────────────────────────────────────────────────
# 3. HARD BOUNDS ENFORCEMENT & INITIALIZATION UTILITIES
# ─────────────────────────────────────────────────────────────────────

def compute_affine_from_correspondences(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    bbox: dict,
    src_shape: Tuple[int, ...],
) -> Optional[np.ndarray]:
    if src_pts is None or ref_pts is None:
        return None
    if len(src_pts) < 3 or len(ref_pts) < 3:
        return None
    try:
        M, inliers = cv2.estimateAffine2D(
            ref_pts.astype(np.float32),
            src_pts.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
        if M is None:
            return None
        return M
    except Exception:
        return None


def compute_homography_from_correspondences(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
) -> Optional[np.ndarray]:
    if src_pts is None or ref_pts is None:
        return None
    if len(src_pts) < 4 or len(ref_pts) < 4:
        return None
    try:
        H, mask = cv2.findHomography(
            ref_pts.astype(np.float32),
            src_pts.astype(np.float32),
            cv2.RANSAC,
            5.0,
        )
        if H is None:
            return None
        det = np.linalg.det(H[:2, :2])
        if det < 0.01 or det > 100.0:
            return None
        return H
    except Exception:
        return None


def displacement_from_mapping(
    mapping_func: Callable[[np.ndarray], np.ndarray],
    bbox: dict,
    mh: int,
    mw: int,
    src_shape: Tuple[int, ...],
    device: torch.device,
    max_disp: float = 0.8,
) -> torch.Tensor:
    src_h, src_w = src_shape[:2]
    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]

    gx = np.linspace(bx0, bx1, mw, dtype=np.float64)
    gy = np.linspace(by0, by1, mh, dtype=np.float64)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    ref_grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    src_grid_pts = mapping_func(ref_grid_pts)

    src_norm_x = (src_grid_pts[:, 0] / (src_w - 1)) * 2.0 - 1.0
    src_norm_y = (src_grid_pts[:, 1] / (src_h - 1)) * 2.0 - 1.0

    src_norm_x = np.clip(src_norm_x, -1.0, 1.0)
    src_norm_y = np.clip(src_norm_y, -1.0, 1.0)

    id_x = np.linspace(-1.0, 1.0, mw, dtype=np.float64)
    id_y = np.linspace(-1.0, 1.0, mh, dtype=np.float64)
    IDX, IDY = np.meshgrid(id_x, id_y, indexing="xy")

    disp_x = src_norm_x.reshape(mh, mw) - IDX
    disp_y = src_norm_y.reshape(mh, mw) - IDY

    disp_mag = np.sqrt(disp_x**2 + disp_y**2)
    scale = np.where(
        disp_mag > max_disp,
        max_disp / np.maximum(disp_mag, 1e-8),
        1.0,
    )
    disp_x *= scale
    disp_y *= scale

    disp_np = np.stack([disp_x, disp_y], axis=0)
    return torch.from_numpy(disp_np).float().unsqueeze(0).to(device)


def validate_displacement(
    disp: torch.Tensor,
    poly_mask_mesh: torch.Tensor,
    src_mask_t: torch.Tensor,
    work_h: int,
    work_w: int,
    ref_mask_t: torch.Tensor,
    poly_work_t: torch.Tensor,
    min_coverage: float = 0.05,
    max_oob_ratio: float = 0.3,
) -> Tuple[bool, float, float, str]:
    device = disp.device
    with torch.no_grad():
        disp_up = F.interpolate(
            disp,
            size=(work_h, work_w),
            mode="bicubic",
            align_corners=True,
        )
        identity = make_identity_grid(work_h, work_w, device)
        grid = identity + disp_up.permute(0, 2, 3, 1)

        oob = (
            (grid[..., 0] < -1)
            | (grid[..., 0] > 1)
            | (grid[..., 1] < -1)
            | (grid[..., 1] > 1)
        )
        oob_ratio = oob.float().mean().item()

        grid_clamped = grid.clone()
        grid_clamped[..., 0] = grid_clamped[..., 0].clamp(-1, 1)
        grid_clamped[..., 1] = grid_clamped[..., 1].clamp(-1, 1)

        warped = F.grid_sample(
            src_mask_t,
            grid_clamped,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped = warped * poly_work_t

        iou = polygon_iou_scalar(warped, ref_mask_t, poly_work_t)
        mean_mag = disp.abs().mean().item()

    reason = "OK"
    is_valid = True

    if oob_ratio > max_oob_ratio:
        reason = f"OOB ratio {oob_ratio:.2%} > {max_oob_ratio:.2%}"
        is_valid = False
    elif mean_mag > 1.5:
        reason = f"Mean displacement {mean_mag:.3f} > 1.5 (extreme)"
        is_valid = False
    elif iou < min_coverage:
        reason = f"IoU {iou:.4f} < {min_coverage} (no meaningful overlap)"
        is_valid = False

    return is_valid, iou, mean_mag, reason


# ─────────────────────────────────────────────────────────────────────
# 4. DISPLACEMENT INITIALIZATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────

def init_disp_from_inverse_poly(
    extent_result: ExtentResult,
    bbox: dict,
    mh: int,
    mw: int,
    src_shape: Tuple[int, ...],
    device: torch.device,
    max_disp: float = 0.8,
) -> torch.Tensor:
    if (
        extent_result.ref_pts is not None
        and len(extent_result.ref_pts) >= 6
        and extent_result.warp_coeffs is not None
    ):
        degree = min(extent_result.warp_coeffs.get("degree", 2), 2)
        try:
            inv_warp = fit_polynomial_warp(
                extent_result.ref_pts,
                extent_result.src_pts,
                degree=degree,
            )

            def mapping_func(pts):
                return apply_polynomial_warp(pts, inv_warp)

            return displacement_from_mapping(
                mapping_func,
                bbox,
                mh,
                mw,
                src_shape,
                device,
                max_disp=max_disp,
            )
        except Exception as e:
            print(f"     ⚠ Polynomial inversion failed: {e}")

    return torch.zeros(1, 2, mh, mw, device=device)


def init_disp_from_affine(
    extent_result: ExtentResult,
    bbox: dict,
    mh: int,
    mw: int,
    src_shape: Tuple[int, ...],
    device: torch.device,
    max_disp: float = 0.8,
) -> Optional[torch.Tensor]:
    M = compute_affine_from_correspondences(
        extent_result.src_pts,
        extent_result.ref_pts,
        bbox,
        src_shape,
    )
    if M is None:
        return None

    def mapping_func(pts):
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        return (M @ pts_h.T).T

    return displacement_from_mapping(
        mapping_func, bbox, mh, mw, src_shape, device, max_disp
    )


def init_disp_from_homography(
    extent_result: ExtentResult,
    bbox: dict,
    mh: int,
    mw: int,
    src_shape: Tuple[int, ...],
    device: torch.device,
    max_disp: float = 0.8,
) -> Optional[torch.Tensor]:
    H = compute_homography_from_correspondences(
        extent_result.src_pts,
        extent_result.ref_pts,
    )
    if H is None:
        return None

    def mapping_func(pts):
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        proj = (H @ pts_h.T).T
        proj[:, 0] /= np.maximum(np.abs(proj[:, 2]), 1e-8)
        proj[:, 1] /= np.maximum(np.abs(proj[:, 2]), 1e-8)
        return proj[:, :2]

    return displacement_from_mapping(
        mapping_func, bbox, mh, mw, src_shape, device, max_disp
    )


def init_displacement_with_validation(
    extent_result: ExtentResult,
    bbox: dict,
    mh: int,
    mw: int,
    src_shape: Tuple[int, ...],
    device: torch.device,
    src_mask_t: torch.Tensor,
    ref_mask_t: torch.Tensor,
    poly_work_t: torch.Tensor,
    work_h: int,
    work_w: int,
    max_disp: float = 0.8,
) -> Tuple[torch.Tensor, str]:
    strategies = []

    d_poly = init_disp_from_inverse_poly(
        extent_result, bbox, mh, mw, src_shape, device, max_disp
    )
    strategies.append(("inverse_poly", d_poly))

    d_homo = init_disp_from_homography(
        extent_result, bbox, mh, mw, src_shape, device, max_disp
    )
    if d_homo is not None:
        strategies.append(("homography", d_homo))

    d_affine = init_disp_from_affine(
        extent_result, bbox, mh, mw, src_shape, device, max_disp
    )
    if d_affine is not None:
        strategies.append(("affine", d_affine))

    d_zero = torch.zeros(1, 2, mh, mw, device=device)
    strategies.append(("identity", d_zero))

    best_disp = d_zero
    best_name = "identity"
    best_iou = -1.0

    poly_mesh = polygon_mask_tensor(
        extent_result.polygon,
        mh,
        mw,
        offset_xy=(bbox["x0"], bbox["y0"]),
        scale_xy=(
            mw / (bbox["x1"] - bbox["x0"]),
            mh / (bbox["y1"] - bbox["y0"]),
        ),
        device=device,
    )

    for name, disp in strategies:
        is_valid, iou, mean_mag, reason = validate_displacement(
            disp,
            poly_mesh,
            src_mask_t,
            work_h,
            work_w,
            ref_mask_t,
            poly_work_t,
            min_coverage=0.01,
            max_oob_ratio=0.4,
        )
        status = "✓" if is_valid else "✗"
        print(
            f"     {status} Strategy '{name}': "
            f"IoU={iou:.4f}  |disp|={mean_mag:.4f}  "
            f"{'VALID' if is_valid else reason}"
        )
        if is_valid and iou > best_iou:
            best_iou = iou
            best_disp = disp
            best_name = name

    if best_iou < 0:
        print("     ⚠ No valid strategy found, using identity")
        best_disp = d_zero
        best_name = "identity (forced)"

    print(f"     → Selected: '{best_name}' (IoU={best_iou:.4f})")
    return best_disp, best_name


# ─────────────────────────────────────────────────────────────────────
# 5. HARD CONSTRAINT PROJECTIONS
# ─────────────────────────────────────────────────────────────────────

def clamp_displacement_to_valid_grid(
    disp: torch.Tensor,
    max_disp_magnitude: float = 0.9,
) -> torch.Tensor:
    mh, mw = disp.shape[2], disp.shape[3]
    device = disp.device

    id_x = torch.linspace(-1, 1, mw, device=device)
    id_y = torch.linspace(-1, 1, mh, device=device)
    IDY, IDX = torch.meshgrid(id_y, id_x, indexing="ij")

    d = disp.clone()
    d[:, 0] = d[:, 0].clamp(
        min=-1.0 - IDX.unsqueeze(0), max=1.0 - IDX.unsqueeze(0)
    )
    d[:, 1] = d[:, 1].clamp(
        min=-1.0 - IDY.unsqueeze(0), max=1.0 - IDY.unsqueeze(0)
    )

    mag = torch.sqrt(d[:, 0:1] ** 2 + d[:, 1:2] ** 2)
    scale = torch.where(
        mag > max_disp_magnitude,
        max_disp_magnitude / mag.clamp(min=1e-8),
        torch.ones_like(mag),
    )
    d = d * scale
    return d


def project_displacement_inplace(
    residual: torch.Tensor,
    frozen_base: torch.Tensor,
    max_disp_magnitude: float = 0.9,
):
    with torch.no_grad():
        total = frozen_base + residual.data
        clamped = clamp_displacement_to_valid_grid(
            total, max_disp_magnitude
        )
        residual.data.copy_(clamped - frozen_base)


# ─────────────────────────────────────────────────────────────────────
# 6. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def dice_loss_poly(
    pred: torch.Tensor,
    target: torch.Tensor,
    poly_mask: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    p = (pred * poly_mask).reshape(-1)
    t = (target * poly_mask).reshape(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def mse_loss_poly(
    pred: torch.Tensor,
    target: torch.Tensor,
    poly_mask: torch.Tensor,
) -> torch.Tensor:
    n = poly_mask.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * poly_mask).sum() / n


def bending_energy_loss(
    disp: torch.Tensor,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    d2x = disp[:, :, :, 2:] - 2 * disp[:, :, :, 1:-1] + disp[:, :, :, :-2]
    d2y = disp[:, :, 2:, :] - 2 * disp[:, :, 1:-1, :] + disp[:, :, :-2, :]
    dxy = (
        disp[:, :, 1:, 1:]
        - disp[:, :, 1:, :-1]
        - disp[:, :, :-1, 1:]
        + disp[:, :, :-1, :-1]
    )

    if weight_map is not None:
        wx = weight_map[:, :, :, 1:-1]
        wy = weight_map[:, :, 1:-1, :]
        wxy = (
            weight_map[:, :, 1:, 1:]
            + weight_map[:, :, 1:, :-1]
            + weight_map[:, :, :-1, 1:]
            + weight_map[:, :, :-1, :-1]
        ) / 4.0
        return (
            (wx * d2x.pow(2)).mean()
            + (wy * d2y.pow(2)).mean()
            + 2.0 * (wxy * dxy.pow(2)).mean()
        )

    return d2x.pow(2).mean() + d2y.pow(2).mean() + 2.0 * dxy.pow(2).mean()


def fold_loss(grid: torch.Tensor) -> torch.Tensor:
    dg_dc_x = grid[:, :, 1:, 0] - grid[:, :, :-1, 0]
    dg_dc_y = grid[:, :, 1:, 1] - grid[:, :, :-1, 1]
    dg_dr_x = grid[:, 1:, :, 0] - grid[:, :-1, :, 0]
    dg_dr_y = grid[:, 1:, :, 1] - grid[:, :-1, :, 1]
    dg_dc_x = dg_dc_x[:, :-1, :]
    dg_dc_y = dg_dc_y[:, :-1, :]
    dg_dr_x = dg_dr_x[:, :, :-1]
    dg_dr_y = dg_dr_y[:, :, :-1]
    det = dg_dc_x * dg_dr_y - dg_dc_y * dg_dr_x
    return F.relu(-det + 0.05).mean()


def jacobian_regularity_loss(grid: torch.Tensor) -> torch.Tensor:
    dg_dc_x = grid[:, :, 1:, 0] - grid[:, :, :-1, 0]
    dg_dc_y = grid[:, :, 1:, 1] - grid[:, :, :-1, 1]
    dg_dr_x = grid[:, 1:, :, 0] - grid[:, :-1, :, 0]
    dg_dr_y = grid[:, 1:, :, 1] - grid[:, :-1, :, 1]
    dg_dc_x = dg_dc_x[:, :-1, :]
    dg_dc_y = dg_dc_y[:, :-1, :]
    dg_dr_x = dg_dr_x[:, :, :-1]
    dg_dr_y = dg_dr_y[:, :, :-1]
    det = dg_dc_x * dg_dr_y - dg_dc_y * dg_dr_x
    mean_det = det.mean().detach()
    return (det - mean_det).pow(2).mean()


def boundary_leakage_loss(
    warped: torch.Tensor,
    poly_mask: torch.Tensor,
) -> torch.Tensor:
    outside = 1.0 - poly_mask
    leakage = (warped * outside).abs()
    return leakage.mean()


# ─────────────────────────────────────────────────────────────────────
# 7. WARPING & GRID HELPERS
# ─────────────────────────────────────────────────────────────────────

def make_identity_grid(
    h: int, w: int, device: torch.device
) -> torch.Tensor:
    yy = torch.linspace(-1, 1, h, device=device)
    xx = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def warp_src_into_bbox(
    src_tensor: torch.Tensor,
    disp: torch.Tensor,
    out_h: int,
    out_w: int,
    poly_mask: Optional[torch.Tensor] = None,
    clamp_grid: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = src_tensor.device
    disp_up = F.interpolate(
        disp,
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=True,
    )
    identity = make_identity_grid(out_h, out_w, device)
    grid = identity.clone()
    grid[..., 0] = grid[..., 0] + disp_up[:, 0]
    grid[..., 1] = grid[..., 1] + disp_up[:, 1]

    if clamp_grid:
        grid = grid.clamp(-1.0, 1.0)

    warped = F.grid_sample(
        src_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    if poly_mask is not None:
        if poly_mask.shape[2] != out_h or poly_mask.shape[3] != out_w:
            poly_mask = F.interpolate(
                poly_mask,
                (out_h, out_w),
                mode="bilinear",
                align_corners=True,
            ).clamp(0, 1)
        warped = warped * poly_mask

    return warped, grid


# ─────────────────────────────────────────────────────────────────────
# 8. METRICS & ADAPTIVE WEIGHT MAPS
# ─────────────────────────────────────────────────────────────────────

def compute_cell_iou_poly(
    warped_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    poly_mask: torch.Tensor,
    mh: int,
    mw: int,
) -> torch.Tensor:
    _, _, H, W = warped_mask.shape
    cell_h = H // mh
    cell_w = W // mw
    iou_map = torch.full((mh, mw), -1.0)

    for i in range(mh):
        for j in range(mw):
            r0 = i * cell_h
            r1 = (i + 1) * cell_h if i < mh - 1 else H
            c0 = j * cell_w
            c1 = (j + 1) * cell_w if j < mw - 1 else W

            pm = poly_mask[0, 0, r0:r1, c0:c1] > 0.5
            if pm.sum() < 1:
                continue

            w_cell = (warped_mask[0, 0, r0:r1, c0:c1] > 0.5) & pm
            r_cell = (ref_mask[0, 0, r0:r1, c0:c1] > 0.5) & pm

            union = (w_cell | r_cell).sum().float()
            if union < 1.0:
                iou_map[i, j] = 1.0
                continue
            inter = (w_cell & r_cell).sum().float()
            iou_map[i, j] = (inter / union).item()

    return iou_map


def polygon_iou_scalar(
    warped_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    poly_mask: torch.Tensor,
) -> float:
    pm = poly_mask > 0.5
    w = (warped_mask > 0.5) & pm
    r = (ref_mask > 0.5) & pm
    inter = (w & r).sum().float()
    union = (w | r).sum().float().clamp(min=1.0)
    return (inter / union).item()


def iou_to_bend_weights(
    cell_iou: torch.Tensor,
    high_w: float = 5.0,
    low_w: float = 0.1,
) -> torch.Tensor:
    weights = torch.full_like(cell_iou, high_w)
    relevant = cell_iou >= 0.0
    weights[relevant] = high_w - (high_w - low_w) * cell_iou[relevant]
    weights[cell_iou < 0.0] = 0.0
    return weights


# ─────────────────────────────────────────────────────────────────────
# 9. VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def show_extent_initialisation(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    extent_result: ExtentResult,
    bbox: dict,
):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB) if len(src_img.shape) == 3 else src_img
    axes[0].imshow(src_rgb, cmap="gray")
    axes[0].set_title(f"Source  {src_img.shape[1]}×{src_img.shape[0]}")
    axes[0].axis("off")

    vis = ref_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if extent_result.polygon is not None:
        pts = extent_result.polygon.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)

    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 200, 255), 2)

    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Extent polygon (green) + tight bbox (cyan)\n"
        f"conf={extent_result.confidence:.3f}  "
        f"{extent_result.n_correspondences} inliers"
    )
    axes[1].axis("off")

    vis2 = ref_img.copy()
    if len(vis2.shape) == 2:
        vis2 = cv2.cvtColor(vis2, cv2.COLOR_GRAY2BGR)
    n_show = 0
    if extent_result.ref_pts is not None and len(extent_result.ref_pts) > 0:
        n_show = min(200, len(extent_result.ref_pts))
        idx = np.random.choice(len(extent_result.ref_pts), n_show, replace=False)
        for i in idx:
            rx = int(extent_result.ref_pts[i, 0])
            ry = int(extent_result.ref_pts[i, 1])
            cv2.circle(vis2, (rx, ry), 4, (0, 0, 255), -1)
        if extent_result.polygon is not None:
            pts = extent_result.polygon.astype(np.int32)
            cv2.polylines(vis2, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    axes[2].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"Inlier correspondences ({n_show} shown)" if n_show > 0 else "No correspondences"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_mask_comparison_poly(
    warped_mask: torch.Tensor,
    ref_mask: torch.Tensor,
    poly_mask: torch.Tensor,
    title: str = "",
):
    wm = warped_mask[0, 0].cpu().numpy()
    rm = ref_mask[0, 0].cpu().numpy()
    pm = poly_mask[0, 0].cpu().numpy() > 0.5

    fig, axes = plt.subplots(1, 4, figsize=(28, 5))

    axes[0].imshow(wm, cmap="gray")
    axes[0].set_title("Warped Source Mask")

    axes[1].imshow(rm, cmap="gray")
    axes[1].set_title("Reference Mask (bbox crop)")

    axes[2].imshow(pm, cmap="gray")
    axes[2].set_title("Polygon Mask")

    rgb = np.zeros((*rm.shape, 3))
    hit = (wm > 0.5) & (rm > 0.5) & pm
    miss = (rm > 0.5) & (wm <= 0.5) & pm
    extra = (wm > 0.5) & (rm <= 0.5) & pm
    rgb[hit, 1] = 1.0
    rgb[miss, 0] = 1.0
    rgb[extra, 2] = 1.0
    axes[3].imshow(rgb)
    axes[3].set_title("Within polygon: G=hit R=miss B=extra")

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_mesh(
    grid_tensor: torch.Tensor,
    ref_img_np: np.ndarray,
    step: int = 8,
    title: str = "Mesh",
):
    g = grid_tensor[0].cpu().numpy()
    h, w = g.shape[:2]
    gx = (g[..., 0] + 1) / 2 * w
    gy = (g[..., 1] + 1) / 2 * h
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(ref_img_np)
    for r in range(0, h, step):
        ax.plot(gx[r, :], gy[r, :], "c-", lw=0.4, alpha=0.7)
    for c in range(0, w, step):
        ax.plot(gx[:, c], gy[:, c], "c-", lw=0.4, alpha=0.7)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_adaptive_maps_poly(
    cell_iou: torch.Tensor,
    bend_weights: torch.Tensor,
    lvl: int,
    mh: int,
    mw: int,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iou_display = cell_iou.cpu().numpy().copy()
    iou_display[iou_display < 0] = np.nan
    im0 = axes[0].imshow(
        iou_display,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_title(
        f"Per-Cell IoU (within polygon)  Level {lvl}\n"
        f"mesh {mw}×{mh}  |  grey = outside polygon"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        bend_weights.cpu().numpy(),
        cmap="hot",
        vmin=0,
        interpolation="nearest",
    )
    axes[1].set_title(
        "Adaptive Bending Weight\n"
        "dark = relax  |  bright = stiffen  |  black = outside polygon"
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    for ax in axes:
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def show_displacement_field(
    total_disp: torch.Tensor,
    lvl: Any,
    mh: int,
    mw: int,
):
    d = total_disp[0].cpu().numpy()
    mag = np.sqrt(d[0] ** 2 + d[1] ** 2)

    grad_y = np.gradient(mag, axis=0)
    grad_x = np.gradient(mag, axis=1)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    im0 = axes[0].imshow(mag, cmap="magma", interpolation="bilinear")
    axes[0].set_title(f"Displacement Magnitude (Level {lvl})\nmesh {mw}×{mh}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    step_q = max(1, min(mh, mw) // 16)
    yy = np.arange(0, mh, step_q)
    xx = np.arange(0, mw, step_q)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    U = d[0][::step_q, ::step_q]
    V = d[1][::step_q, ::step_q]
    axes[1].imshow(mag, cmap="magma", interpolation="bilinear", alpha=0.4)
    axes[1].quiver(
        X, Y, U, V, mag[::step_q, ::step_q], cmap="coolwarm", scale=None, width=0.004
    )
    axes[1].set_title("Displacement Vectors")

    im2 = axes[2].imshow(grad_mag, cmap="inferno", interpolation="bilinear")
    axes[2].set_title(
        "‖∇ magnitude‖  (Spatial Continuity)\nSmooth = good  |  Sharp edges = bad"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_initialisation_comparison(
    src_mask_t: torch.Tensor,
    ref_mask_t: torch.Tensor,
    poly_work_t: torch.Tensor,
    disp: torch.Tensor,
    work_h: int,
    work_w: int,
    strategy_name: str,
):
    with torch.no_grad():
        warped, grid = warp_src_into_bbox(
            src_mask_t,
            disp,
            work_h,
            work_w,
            poly_mask=poly_work_t,
        )
    show_mask_comparison_poly(
        warped,
        ref_mask_t,
        poly_work_t,
        title=f"INITIAL WARP (before optimisation) — strategy: {strategy_name}",
    )


# ─────────────────────────────────────────────────────────────────────
# 10. MAIN SGD MESH WARP PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────

def sgd_mesh_warp_polygon_constrained(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    extent_result: ExtentResult,
    work_h_bbox: int = 384,
    cell_schedule: Tuple[int, ...] = (48, 24, 12, 6, 4, 2),
    steps_per_lvl: Tuple[int, ...] = (600, 600, 600, 600, 800, 800),
    lr_schedule: Tuple[float, ...] = (5e-3, 3e-3, 1.5e-3, 8e-4, 3e-4, 1e-4),
    lam_bend_schedule: Tuple[float, ...] = (0.5, 1.0, 2.0, 6.0, 15.0, 40.0),
    lam_residual: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.08, 0.2, 0.4),
    lam_fold: float = 0.2,
    lam_jac: float = 0.1,
    lam_leakage: float = 5.0,
    proj_sigma_schedule: Tuple[float, ...] = (2.0, 1.5, 1.0, 0.6, 0.35, 0.2),
    proj_every: int = 50,
    max_disp_schedule: Tuple[float, ...] = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85),
    hard_clamp_every: int = 10,
) -> Tuple[np.ndarray, torch.Tensor, dict, float]:
    if extent_result.polygon is None:
        raise ValueError("extent_result.polygon is None — run find_extent() first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    h_r, w_r = ref_img.shape[:2]
    h_s, w_s = src_img.shape[:2]
    print(f"Reference : {w_r}×{h_r}")
    print(f"Source    : {w_s}×{h_s}")

    # ── Phase 1: Derive spatial constraints ────────────────────────
    print("\n▸ Phase 1: Deriving constraints from ExtentResult …")
    polygon = extent_result.polygon
    bbox = polygon_tight_bbox(polygon, ref_img.shape)
    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    bw = bx1 - bx0
    bh = by1 - by0
    print(f"  Polygon bbox: ({bx0},{by0})→({bx1},{by1})  size {bw}×{bh}")

    show_extent_initialisation(ref_img, src_img, extent_result, bbox)

    # ── Phase 2: Land masks ────────────────────────────────────────
    print("\n▸ Phase 2: Extracting land masks …")
    mask_ref_full = extract_land_mask(ref_img)
    mask_src_full = extract_land_mask(src_img)
    print(
        f"  ref land: {(mask_ref_full > 0).mean() * 100:.1f}%  "
        f"src land: {(mask_src_full > 0).mean() * 100:.1f}%"
    )

    # ── Phase 3: Build tensors ─────────────────────────────────────
    print("\n▸ Phase 3: Building tensors …")

    COARSEST_CELL = cell_schedule[0]
    raw_w = work_h_bbox * bw / bh
    work_w_bbox = max(
        COARSEST_CELL,
        int(round(raw_w / COARSEST_CELL)) * COARSEST_CELL,
    )
    print(f"  Bbox working resolution: {work_w_bbox}×{work_h_bbox}")

    scale_x_work = work_w_bbox / bw
    scale_y_work = work_h_bbox / bh

    poly_mask_work_np = rasterise_polygon_mask(
        polygon,
        work_h_bbox,
        work_w_bbox,
        offset_xy=(bx0, by0),
        scale_xy=(scale_x_work, scale_y_work),
    )
    poly_mask_work_t = (
        (torch.from_numpy(poly_mask_work_np).float() / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    ref_bbox_mask_np = mask_ref_full[by0:by1, bx0:bx1]
    ref_bbox_t_full = (
        (torch.from_numpy(ref_bbox_mask_np).float() / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    ref_bbox_sm = F.interpolate(
        ref_bbox_t_full,
        (work_h_bbox, work_w_bbox),
        mode="bilinear",
        align_corners=True,
    )

    src_mask_t = (
        (torch.from_numpy(mask_src_full).float() / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    src_sm = F.interpolate(
        src_mask_t,
        (work_h_bbox, work_w_bbox),
        mode="bilinear",
        align_corners=True,
    )

    src_rgb_np = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img_t = (
        torch.from_numpy(src_rgb_np)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        / 255.0
    )

    # ── Phase 3b: Validated Initialisation ─────────────────────────
    print("\n▸ Phase 3b: Multi-strategy initialisation with validation …")

    init_mh = work_h_bbox // COARSEST_CELL
    init_mw = work_w_bbox // COARSEST_CELL
    max_disp_init = max_disp_schedule[0]

    init_disp, init_strategy = init_displacement_with_validation(
        extent_result=extent_result,
        bbox=bbox,
        mh=init_mh,
        mw=init_mw,
        src_shape=src_img.shape,
        device=device,
        src_mask_t=src_sm,
        ref_mask_t=ref_bbox_sm,
        poly_work_t=poly_mask_work_t,
        work_h=work_h_bbox,
        work_w=work_w_bbox,
        max_disp=max_disp_init,
    )

    show_initialisation_comparison(
        src_sm,
        ref_bbox_sm,
        poly_mask_work_t,
        init_disp,
        work_h_bbox,
        work_w_bbox,
        init_strategy,
    )

    # ── Phase 4: Coarse-to-fine SGD ───────────────────────────────
    print(
        "\n▸ Phase 4: Polygon-Constrained Coarse-to-Fine "
        "Optimisation (hard-bounded) …"
    )
    print(
        f"  Cell sizes (px): {list(cell_schedule)}  →  finest mesh = "
        f"{work_w_bbox // cell_schedule[-1]}×{work_h_bbox // cell_schedule[-1]}"
    )

    base_disp = init_disp.clone()
    bend_weight_map = None

    for lvl, (cell_sz, steps, lr) in enumerate(
        zip(cell_schedule, steps_per_lvl, lr_schedule)
    ):
        mh = work_h_bbox // cell_sz
        mw = work_w_bbox // cell_sz
        if mh < 2 or mw < 2:
            print(f"  Skipping cell={cell_sz}: mesh {mw}×{mh} too small")
            continue

        lam_bend = lam_bend_schedule[lvl]
        lam_res = lam_residual[lvl]
        proj_sigma = proj_sigma_schedule[lvl]
        max_disp = max_disp_schedule[min(lvl, len(max_disp_schedule) - 1)]

        print(
            f"\n  ── Level {lvl + 1}  cell={cell_sz}px  mesh {mw}×{mh}  "
            f"steps={steps}  lr={lr}  λ_bend={lam_bend}  λ_res={lam_res}  "
            f"max_disp={max_disp}"
        )

        if lvl == 0:
            frozen_base = base_disp.detach()
            frozen_base = clamp_displacement_to_valid_grid(frozen_base, max_disp)
            print(
                f"     Seeded from '{init_strategy}'  "
                f"|disp|_mean={frozen_base.abs().mean().item():.4f}"
            )
        else:
            frozen_base = F.interpolate(
                base_disp.detach(),
                (mh, mw),
                mode="bicubic",
                align_corners=True,
            )
            frozen_base = gaussian_blur_2d(frozen_base, sigma=1.0)
            frozen_base = clamp_displacement_to_valid_grid(frozen_base, max_disp)

        if lvl == 0:
            residual = torch.zeros(1, 2, mh, mw, device=device, requires_grad=True)
        else:
            prev_total = F.interpolate(
                base_disp.detach(),
                (mh, mw),
                mode="bicubic",
                align_corners=True,
            )
            init_res = prev_total - frozen_base
            init_res = gaussian_blur_2d(init_res, sigma=proj_sigma)
            total_check = frozen_base + init_res
            total_clamped = clamp_displacement_to_valid_grid(total_check, max_disp)
            init_res = total_clamped - frozen_base
            residual = init_res.clone().requires_grad_(True)

        if bend_weight_map is None:
            poly_mesh_mask_np = rasterise_polygon_mask(
                polygon,
                mh,
                mw,
                offset_xy=(bx0, by0),
                scale_xy=(mw / bw, mh / bh),
            )
            bend_w = (
                torch.from_numpy(poly_mesh_mask_np).float() / 255.0
            ).unsqueeze(0).unsqueeze(0).to(device) * 2.0
        else:
            bend_w = F.interpolate(
                bend_weight_map,
                (mh, mw),
                mode="bilinear",
                align_corners=True,
            )

        print(
            f"     Bending weights  min={bend_w.min().item():.2f}  "
            f"max={bend_w.max().item():.2f}  mean={bend_w.mean().item():.2f}"
        )

        poly_mesh_np = rasterise_polygon_mask(
            polygon,
            mh * cell_sz,
            mw * cell_sz,
            offset_xy=(bx0, by0),
            scale_xy=(mw * cell_sz / bw, mh * cell_sz / bh),
        )
        poly_work_t = (
            torch.from_numpy(poly_mesh_np).float() / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)
        poly_work_t = F.interpolate(
            poly_work_t,
            (work_h_bbox, work_w_bbox),
            mode="bilinear",
            align_corners=True,
        ).clamp(0, 1)

        optim = torch.optim.Adam([residual], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps)

        best_loss = float("inf")
        best_residual = residual.data.clone()
        losses_log = []

        for step in range(steps):
            optim.zero_grad()

            total_disp = frozen_base + residual

            warped, grid = warp_src_into_bbox(
                src_sm,
                total_disp,
                work_h_bbox,
                work_w_bbox,
                poly_mask=poly_work_t,
                clamp_grid=True,
            )

            l_dice = dice_loss_poly(warped, ref_bbox_sm, poly_work_t)
            l_mse = mse_loss_poly(warped, ref_bbox_sm, poly_work_t)
            l_bend = bending_energy_loss(total_disp, bend_w)
            l_fold = fold_loss(grid)
            l_jac = jacobian_regularity_loss(grid)
            l_res_mag = residual.pow(2).mean()

            warped_raw = F.grid_sample(
                src_sm,
                grid.clamp(-1, 1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            l_leak = boundary_leakage_loss(warped_raw, poly_work_t)

            loss = (
                l_dice
                + l_mse
                + lam_bend * l_bend
                + lam_fold * l_fold
                + lam_jac * l_jac
                + lam_res * l_res_mag
                + lam_leakage * l_leak
            )
            loss.backward()
            optim.step()
            sched.step()

            losses_log.append(loss.item())

            if (step + 1) % hard_clamp_every == 0:
                project_displacement_inplace(residual, frozen_base, max_disp)

            if (step + 1) % proj_every == 0 and proj_sigma > 0.1:
                with torch.no_grad():
                    smoothed = gaussian_blur_2d(residual.data, sigma=proj_sigma)
                    residual.data.copy_(smoothed)
                    project_displacement_inplace(residual, frozen_base, max_disp)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_residual = residual.data.clone()

            if step % 300 == 0 or step == steps - 1:
                with torch.no_grad():
                    td = frozen_base + residual
                    wc, _ = warp_src_into_bbox(
                        src_sm,
                        td,
                        work_h_bbox,
                        work_w_bbox,
                        poly_mask=poly_work_t,
                    )
                    iou = polygon_iou_scalar(wc, ref_bbox_sm, poly_work_t)

                    td_up = F.interpolate(
                        td,
                        (work_h_bbox, work_w_bbox),
                        mode="bicubic",
                        align_corners=True,
                    )
                    id_grid = make_identity_grid(work_h_bbox, work_w_bbox, device)
                    check_grid = id_grid.clone()
                    check_grid[..., 0] += td_up[:, 0]
                    check_grid[..., 1] += td_up[:, 1]
                    oob_frac = (check_grid.abs() > 1.0).float().mean().item()

                print(
                    f"     step {step:5d}  loss={loss.item():.5f}  "
                    f"dice={l_dice.item():.4f}  bend={l_bend.item():.4f}  "
                    f"leak={l_leak.item():.4f}  polygon_IoU={iou:.4f}  "
                    f"OOB={oob_frac:.4f}"
                )

        best_total = frozen_base + best_residual
        best_total = clamp_displacement_to_valid_grid(best_total, max_disp)
        base_disp = gaussian_blur_2d(best_total, sigma=proj_sigma * 0.5)
        base_disp = clamp_displacement_to_valid_grid(base_disp, max_disp)

        with torch.no_grad():
            wv, _ = warp_src_into_bbox(
                src_sm,
                base_disp,
                work_h_bbox,
                work_w_bbox,
                poly_mask=poly_work_t,
            )
            cell_iou = compute_cell_iou_poly(wv, ref_bbox_sm, poly_work_t, mh, mw)
            raw_bw = iou_to_bend_weights(cell_iou, high_w=5.0, low_w=0.1)
            bend_weight_map = raw_bw.unsqueeze(0).unsqueeze(0).to(device)

        show_mask_comparison_poly(
            wv,
            ref_bbox_sm,
            poly_work_t,
            title=f"After Level {lvl + 1} (cell {cell_sz}px → mesh {mw}×{mh})",
        )
        show_adaptive_maps_poly(cell_iou, raw_bw, lvl + 1, mh, mw)
        show_displacement_field(base_disp, lvl + 1, mh, mw)

        plt.figure(figsize=(8, 3))
        plt.plot(losses_log, linewidth=0.8)
        plt.title(f"Level {lvl + 1} loss curve")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    current_disp = base_disp

    # ── Phase 5: Full-resolution warp within bbox ──────────────────
    print("\n▸ Phase 5: Full-resolution warp within bbox …")

    poly_mask_bbox_full_np = rasterise_polygon_mask(
        polygon,
        bh,
        bw,
        offset_xy=(bx0, by0),
        scale_xy=(bw / bw, bh / bh),
    )
    poly_mask_bbox_full = (
        torch.from_numpy(poly_mask_bbox_full_np).float() / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        disp_full = F.interpolate(
            current_disp,
            size=(bh, bw),
            mode="bicubic",
            align_corners=True,
        )
        identity_bbox = make_identity_grid(bh, bw, device)
        grid_bbox = identity_bbox.clone()
        grid_bbox[..., 0] = grid_bbox[..., 0] + disp_full[:, 0]
        grid_bbox[..., 1] = grid_bbox[..., 1] + disp_full[:, 1]

        grid_bbox = grid_bbox.clamp(-1.0, 1.0)

        warped_color_bbox = F.grid_sample(
            src_img_t,
            grid_bbox,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_mask_bbox = F.grid_sample(
            src_mask_t,
            grid_bbox,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        warped_color_bbox = warped_color_bbox * poly_mask_bbox_full
        warped_mask_bbox = warped_mask_bbox * poly_mask_bbox_full

    sea_color = get_sea_color(ref_img)
    canvas_np = np.full_like(ref_img, sea_color)

    warped_bbox_np = (
        warped_color_bbox[0].permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    warped_bbox_bgr = cv2.cvtColor(warped_bbox_np, cv2.COLOR_RGB2BGR)

    poly_hw = poly_mask_bbox_full_np > 127
    region = canvas_np[by0:by1, bx0:bx1]
    region[poly_hw] = warped_bbox_bgr[poly_hw]
    canvas_np[by0:by1, bx0:bx1] = region

    mask_canvas = np.zeros((h_r, w_r), dtype=np.float32)
    wm_np = warped_mask_bbox[0, 0].cpu().numpy()
    wm_np_masked = wm_np * (poly_mask_bbox_full_np / 255.0)
    mask_canvas[by0:by1, bx0:bx1] = wm_np_masked

    warped_mask_full_t = (
        torch.from_numpy(mask_canvas).float().unsqueeze(0).unsqueeze(0).to(device)
    )
    ref_mask_full_t = (
        (torch.from_numpy(mask_ref_full).float() / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )
    poly_full_t = (
        (
            torch.from_numpy(rasterise_polygon_mask(polygon, h_r, w_r)).float()
            / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    final_iou = polygon_iou_scalar(
        warped_mask_full_t, ref_mask_full_t, poly_full_t
    )
    print(f"  Final polygon IoU (full res): {final_iou:.4f}")

    with torch.no_grad():
        oob_count = (
            (grid_bbox[..., 0].abs() > 1.001)
            | (grid_bbox[..., 1].abs() > 1.001)
        ).sum().item()
    print(f"  Grid OOB pixels (after clamp): {int(oob_count)} (should be 0)")
    if oob_count > 0:
        print("  ⚠ WARNING: OOB pixels detected after clamping!")

    # ── Phase 6: Final Visualisations ─────────────────────────────
    print("\n▸ Phase 6: Visualisation …")
    ref_rgb_np = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    canvas_rgb = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB)
    src_rgb_orig = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        full_identity = make_identity_grid(h_r, w_r, device)
        full_grid = full_identity.clone()
        full_grid[:, by0:by1, bx0:bx1, :] = grid_bbox
    show_mesh(
        full_grid,
        ref_rgb_np,
        step=max(1, h_r // 80),
        title="Deformed Sampling Mesh (polygon-constrained, v2)",
    )

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas_rgb)
    plt.title(f"Warped Source → Reference frame (polygon IoU = {final_iou:.4f})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    poly_mask_vis = rasterise_polygon_mask(polygon, h_r, w_r)
    overlay = cv2.addWeighted(ref_img, 0.45, canvas_np, 0.55, 0)
    overlay[poly_mask_vis == 0] = ref_img[poly_mask_vis == 0]
    cv2.polylines(
        overlay,
        [polygon.astype(np.int32)],
        True,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay: warped source blended inside polygon, reference outside")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    ref_bbox_full_t = (
        torch.from_numpy(mask_ref_full[by0:by1, bx0:bx1]).float() / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)
    warp_bbox_full_t = (
        torch.from_numpy(mask_canvas[by0:by1, bx0:bx1]).float()
    ).unsqueeze(0).unsqueeze(0).to(device)
    show_mask_comparison_poly(
        warp_bbox_full_t,
        ref_bbox_full_t,
        poly_mask_bbox_full,
        title=f"Final Full-Resolution Overlap  IoU={final_iou:.4f}",
    )

    ref_crop_rgb = cv2.cvtColor(ref_img[by0:by1, bx0:bx1], cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    axes[0].imshow(src_rgb_orig)
    axes[0].set_title("Original Source")
    axes[1].imshow(cv2.cvtColor(warped_bbox_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Warped Source (bbox crop)")
    axes[2].imshow(ref_crop_rgb)
    axes[2].set_title("Reference (bbox crop)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Bbox Region Comparison  —  polygon IoU = {final_iou:.4f}",
        fontsize=15,
    )
    plt.tight_layout()
    plt.show()

    show_displacement_field(
        current_disp,
        "FINAL",
        current_disp.shape[2],
        current_disp.shape[3],
    )

    print("\n✓ Done.")
    return canvas_np, current_disp, bbox, final_iou