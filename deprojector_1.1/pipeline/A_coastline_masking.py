import cv2
import numpy as np
from typing import Tuple
from scipy.ndimage import binary_fill_holes
from scipy.signal import fftconvolve

# Custom imports
from core.dataclass import NormalizedPair, SegmentationResult

def _dominant_colors_kmeans(
    img_lab: np.ndarray,
    k: int = 5,
    sample_n: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    K-means clustering in Lab space.
    Returns (centers, labels, pixel_counts) sorted by count desc.
    """
    h, w = img_lab.shape[:2]
    pixels = img_lab.reshape(-1, 3).astype(np.float32)

    if len(pixels) > sample_n:
        idx = np.random.choice(len(pixels), sample_n, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30, 1.0,
    )
    _, labels_sample, centers = cv2.kmeans(
        sample, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # Assign all pixels to nearest center
    dists = np.linalg.norm(
        pixels[:, None, :] - centers[None, :, :], axis=2
    )
    labels_full = np.argmin(dists, axis=1)

    unique, counts = np.unique(labels_full, return_counts=True)
    order = np.argsort(-counts)
    centers = centers[order]
    counts = counts[order]
    # Remap labels
    remap = np.zeros(k, dtype=int)
    remap[order] = np.arange(k)
    labels_full = remap[labels_full]

    return centers, labels_full.reshape(h, w), counts


def _classify_sea_clusters(
    centers_lab: np.ndarray,
    counts: np.ndarray,
    img_lab: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Classify each cluster as sea or land.

    Sea heuristics (in Lab space):
    - Tends to be the largest connected region touching image border
    - Usually has lower a* (less red/green), moderate-high b* (blue)
    - Usually more uniform (low local variance)

    Returns: boolean array, True = sea cluster.
    """
    h, w = labels.shape
    k = len(centers_lab)
    scores = np.zeros(k, dtype=np.float64)

    # 1. Border contact score: sea typically touches the border
    border_pixels = np.concatenate([
        labels[0, :], labels[-1, :],
        labels[:, 0], labels[:, -1],
    ])
    border_counts = np.bincount(border_pixels, minlength=k)
    total_border = len(border_pixels)
    border_frac = border_counts / total_border
    scores += border_frac * 3.0

    # 2. Area score: sea is usually the largest region
    area_frac = counts / counts.sum()
    scores += area_frac * 2.0

    # 3. Color uniformity: sea regions tend to be more uniform
    for ci in range(k):
        cluster_mask = labels == ci
        if cluster_mask.sum() < 100:
            continue
        local_var = _compute_local_variance(
            img_lab[:, :, 0], cluster_mask
        )
        # Lower variance → more sea-like
        uniformity = 1.0 / (1.0 + local_var / 50.0)
        scores[ci] += uniformity * 1.5

    # 4. Connected component analysis: sea is usually one big blob
    for ci in range(k):
        cluster_mask = (labels == ci).astype(np.uint8)
        n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
            cluster_mask, connectivity=8
        )
        if n_labels <= 1:
            continue
        # Fraction of cluster in its largest component
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_frac = areas.max() / areas.sum()
        scores[ci] += largest_frac * 1.5

    # 5. Blueness heuristic in Lab: low a*, positive b* offset
    #    This is a soft signal — many maps have non-blue seas
    for ci in range(k):
        L, a, b = centers_lab[ci]
        # Blue in Lab: a is slightly negative, b is negative
        # But map "sea blue" varies wildly, so this is weak
        blue_score = max(0, (128 - b) / 128) * 0.3
        scores[ci] += blue_score

    # Determine threshold: largest gap in sorted scores
    sorted_scores = np.sort(scores)[::-1]
    gaps = np.diff(sorted_scores)
    if len(gaps) > 0:
        best_gap_idx = np.argmin(gaps)  # largest negative gap
        threshold = (
            sorted_scores[best_gap_idx]
            + sorted_scores[best_gap_idx + 1]
        ) / 2
    else:
        threshold = scores.mean()

    is_sea = scores >= threshold

    # Safety: at least one cluster must be sea and one land
    if is_sea.all():
        is_sea[np.argmin(scores)] = False
    if not is_sea.any():
        is_sea[np.argmax(scores)] = True

    return is_sea


def _compute_local_variance(
    channel: np.ndarray,
    mask: np.ndarray,
    ksize: int = 15,
) -> float:
    """Mean local variance of a channel within a mask."""
    ch = channel.astype(np.float64)
    m = mask.astype(np.float64)

    kernel = np.ones((ksize, ksize), dtype=np.float64)
    kernel /= kernel.sum()

    local_mean = fftconvolve(ch * m, kernel, mode='same')
    local_sq = fftconvolve((ch ** 2) * m, kernel, mode='same')
    local_count = fftconvolve(m, kernel, mode='same')
    local_count = np.maximum(local_count, 1)

    local_mean /= local_count
    local_sq /= local_count
    local_var = np.maximum(local_sq - local_mean ** 2, 0)

    valid = mask > 0
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(local_var[valid]))

def _signed_dt_to_uint8(sdt: np.ndarray) -> np.ndarray:
    """Convert float32 signed distance to uint8 for matching."""
    mx = max(np.abs(sdt).max(), 1.0)
    return ((sdt / mx) * 127 + 128).clip(0, 255).astype(np.uint8)


def _coast_dt_to_uint8(
    cdt: np.ndarray, max_dist: float = 35.0
) -> np.ndarray:
    """Convert coastline distance to edge-distance-like uint8."""
    clamped = np.clip(cdt, 0, max_dist)
    return (255 - (clamped / max_dist * 255)).astype(np.uint8)


def _float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize any float image to 0–255 uint8."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-10:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)

# ─── Main Segmentation ──────────────────────────────────────────

def segment_land_sea(
    img: np.ndarray,
    n_clusters: int = 5,
    morph_size: int = 7,
    min_region_frac: float = 0.001,
    verbose: bool = False,
) -> SegmentationResult:
    """
    Robust land/sea segmentation using Lab color clustering,
    border analysis, uniformity, and morphological cleanup.
    """
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    # Slight blur to reduce noise/compression artifacts
    img_lab_smooth = cv2.GaussianBlur(img_lab, (5, 5), 1.5)

    # Cluster
    centers, labels, counts = _dominant_colors_kmeans(
        img_lab_smooth, k=n_clusters
    )

    # Classify clusters
    is_sea = _classify_sea_clusters(centers, counts, img_lab, labels)

    if verbose:
        for i in range(len(centers)):
            tag = "SEA" if is_sea[i] else "LAND"
            print(
                f"  Cluster {i}: Lab=({centers[i][0]:.0f}, "
                f"{centers[i][1]:.0f}, {centers[i][2]:.0f}), "
                f"area={counts[i]/counts.sum()*100:.1f}%, "
                f"→ {tag}"
            )

    # Build raw masks
    sea_raw = np.zeros(labels.shape, dtype=np.uint8)
    for ci in range(len(centers)):
        if is_sea[ci]:
            sea_raw[labels == ci] = 255

    land_raw = 255 - sea_raw

    # Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_size, morph_size)
    )

    # Close small holes in land, then open to remove specks
    land_clean = cv2.morphologyEx(land_raw, cv2.MORPH_CLOSE, kernel)
    land_clean = cv2.morphologyEx(land_clean, cv2.MORPH_OPEN, kernel)

    # Same for sea
    sea_clean = cv2.morphologyEx(sea_raw, cv2.MORPH_CLOSE, kernel)
    sea_clean = cv2.morphologyEx(sea_clean, cv2.MORPH_OPEN, kernel)

    # Fill holes in land regions
    land_filled = (
        binary_fill_holes(land_clean > 127).astype(np.uint8) * 255
    )

    # Remove tiny components
    land_final = _remove_small_components(
        land_filled,
        min_frac=min_region_frac,
    )
    sea_final = 255 - land_final

    # Extract coastline
    coastline = _extract_coastline(land_final, thickness=2)

    # Compute mean colors
    sea_pixels = img_lab[sea_final > 127]
    land_pixels = img_lab[land_final > 127]
    sea_color = (
        sea_pixels.mean(axis=0) if len(sea_pixels) > 0
        else np.array([128, 128, 128])
    )
    land_color = (
        land_pixels.mean(axis=0) if len(land_pixels) > 0
        else np.array([128, 128, 128])
    )

    # Confidence: how bimodal is the distribution?
    sea_frac = (sea_final > 127).mean()
    land_frac = (land_final > 127).mean()
    balance = 1.0 - abs(sea_frac - land_frac)
    color_sep = np.linalg.norm(sea_color - land_color) / 255.0
    confidence = 0.5 * balance + 0.5 * color_sep

    if verbose:
        print(
            f"  Land: {land_frac*100:.1f}%, Sea: {sea_frac*100:.1f}%"
        )
        print(
            f"  Color separation (Lab): {color_sep*255:.1f}"
        )
        print(f"  Segmentation confidence: {confidence:.3f}")

    return SegmentationResult(
        land_mask=land_final,
        sea_mask=sea_final,
        coastline=coastline,
        sea_color_lab=sea_color,
        land_color_lab=land_color,
        confidence=confidence,
        debug={
            "labels": labels,
            "centers_lab": centers,
            "counts": counts,
            "is_sea": is_sea,
            "land_raw": land_raw,
            "sea_raw": sea_raw,
        },
    )


def _remove_small_components(
    mask: np.ndarray,
    min_frac: float = 0.001,
) -> np.ndarray:
    """Remove connected components smaller than min_frac of image."""
    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    total = mask.shape[0] * mask.shape[1]
    min_area = total * min_frac
    out = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labeled == i] = 255
    return out


def _extract_coastline(
    land_mask: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """Extract coastline as the boundary between land and sea."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1)
    )
    dilated = cv2.dilate(land_mask, kernel)
    eroded = cv2.erode(land_mask, kernel)
    coastline = cv2.subtract(dilated, eroded)
    return coastline


# ─── Mutual Normalization ────────────────────────────────────────

def _cross_validate_segmentation(
    ref_seg: SegmentationResult,
    src_seg: SegmentationResult,
    ref_img_lab: np.ndarray,
    src_img_lab: np.ndarray,
    verbose: bool = False,
) -> Tuple[SegmentationResult, SegmentationResult]:
    """
    Ensure both segmentations are classifying land/sea consistently.

    If the sea color of one image is closer to the land color of the
    other, we have an inversion — flip the problematic one.
    """
    ref_sea = ref_seg.sea_color_lab
    ref_land = ref_seg.land_color_lab
    src_sea = src_seg.sea_color_lab
    src_land = src_seg.land_color_lab

    # Distance matrix between ref and src classes
    d_ss = np.linalg.norm(ref_sea - src_sea)      # sea↔sea
    d_ll = np.linalg.norm(ref_land - src_land)    # land↔land
    d_sl = np.linalg.norm(ref_sea - src_land)     # ref_sea↔src_land
    d_ls = np.linalg.norm(ref_land - src_sea)     # ref_land↔src_sea

    # Normal: sea matches sea, land matches land
    normal_cost = d_ss + d_ll
    # Inverted: sea matches land and vice versa
    inverted_cost = d_sl + d_ls

    if verbose:
        print("  Cross-validation:")
        print(f"    Normal cost: {normal_cost:.1f} "
              f"(sea↔sea={d_ss:.1f}, land↔land={d_ll:.1f})")
        print(f"    Inverted cost: {inverted_cost:.1f} "
              f"(sea↔land={d_sl:.1f}, land↔sea={d_ls:.1f})")

    if inverted_cost < normal_cost * 0.8:
        if verbose:
            print("    ⚠ Detected inversion in source — flipping")
        # Flip the source segmentation
        src_seg = SegmentationResult(
            land_mask=src_seg.sea_mask,
            sea_mask=src_seg.land_mask,
            coastline=src_seg.coastline,  # coastline unchanged
            sea_color_lab=src_seg.land_color_lab,
            land_color_lab=src_seg.sea_color_lab,
            confidence=src_seg.confidence,
            debug=src_seg.debug,
        )
    elif verbose:
        print("    ✓ Classifications are consistent")

    return ref_seg, src_seg


def _compute_signed_distance(mask: np.ndarray) -> np.ndarray:
    """
    Signed distance transform: positive inside land, negative in sea.
    Returns float32.
    """
    binary = (mask > 127).astype(np.uint8)
    dt_land = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dt_sea = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    return (dt_land - dt_sea).astype(np.float32)


def _compute_coastline_curvature(
    coastline: np.ndarray,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Compute curvature at each pixel based on coastline orientation.
    High curvature = peninsula tips, bay inlets (distinctive).
    Low curvature = straight coastline (less distinctive).
    Returns float32 image.
    """
    # Find contours in the coastline mask
    contours, _ = cv2.findContours(
        coastline, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    curvature_img = np.zeros(
        coastline.shape, dtype=np.float32
    )

    for contour in contours:
        if len(contour) < 10:
            continue
        pts = contour.squeeze()
        if pts.ndim != 2:
            continue

        # Gaussian-smooth the contour coordinates
        from scipy.ndimage import gaussian_filter1d
        x = gaussian_filter1d(
            pts[:, 0].astype(np.float64), sigma, mode='wrap'
        )
        y = gaussian_filter1d(
            pts[:, 1].astype(np.float64), sigma, mode='wrap'
        )

        # Curvature via first and second derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom = np.maximum(denom, 1e-10)
        kappa = np.abs(dx * ddy - dy * ddx) / denom

        # Paint curvature onto image
        for i in range(len(pts)):
            px, py = int(round(pts[i, 0])), int(round(pts[i, 1]))
            if 0 <= py < curvature_img.shape[0] and \
               0 <= px < curvature_img.shape[1]:
                curvature_img[py, px] = max(
                    curvature_img[py, px], kappa[i]
                )

    # Dilate slightly so it's not single-pixel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    curvature_img = cv2.dilate(curvature_img, kernel)

    return curvature_img


def _compute_coastline_orientation(
    land_mask: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Gradient orientation of the land mask boundary.
    Encodes the direction the coastline faces at each point.
    Returns float32 in [0, pi].
    """
    blurred = cv2.GaussianBlur(
        land_mask.astype(np.float32), (0, 0), sigma
    )
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
    orientation = np.arctan2(gy, gx) % np.pi
    # Mask to coastline vicinity only
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    thresh = magnitude.max() * 0.05
    orientation[magnitude < thresh] = 0
    return orientation.astype(np.float32)


def _build_shape_context_image(
    land_mask: np.ndarray,
    coastline: np.ndarray,
    n_rings: int = 4,
    n_angles: int = 8,
    max_radius: float = 60.0,
) -> np.ndarray:
    """
    Per-pixel shape context descriptor image.

    For each coastline pixel, compute a log-polar histogram of
    nearby coastline pixels. This is distortion-tolerant because
    it captures relative spatial arrangement of coast features.

    Returns (H, W, n_rings*n_angles) float32.
    """
    h, w = coastline.shape
    coast_pts = np.column_stack(np.where(coastline > 127))

    # Subsample for efficiency
    max_pts = 5000
    if len(coast_pts) > max_pts:
        idx = np.random.choice(
            len(coast_pts), max_pts, replace=False
        )
        coast_pts = coast_pts[idx]

    n_bins = n_rings * n_angles
    desc_img = np.zeros((h, w, n_bins), dtype=np.float32)

    if len(coast_pts) < 10:
        return desc_img

    from scipy.spatial import cKDTree
    tree = cKDTree(coast_pts)

    # Log-polar bin edges
    ring_edges = np.logspace(
        np.log10(2), np.log10(max_radius), n_rings + 1
    )
    angle_edges = np.linspace(0, 2 * np.pi, n_angles + 1)

    for i, (py, px) in enumerate(coast_pts):
        neighbors = tree.query_ball_point([py, px], max_radius)
        if len(neighbors) < 2:
            continue

        nbrs = coast_pts[neighbors]
        dy = nbrs[:, 0] - py
        dx = nbrs[:, 1] - px
        dists = np.sqrt(dy ** 2 + dx ** 2)
        angles = np.arctan2(dy, dx) % (2 * np.pi)

        # Skip self
        valid = dists > 0.5
        dists = dists[valid]
        angles = angles[valid]

        if len(dists) == 0:
            continue

        # Bin into log-polar histogram
        ring_idx = np.searchsorted(ring_edges, dists) - 1
        angle_idx = np.searchsorted(angle_edges, angles) - 1
        ring_idx = np.clip(ring_idx, 0, n_rings - 1)
        angle_idx = np.clip(angle_idx, 0, n_angles - 1)

        hist = np.zeros(n_bins, dtype=np.float32)
        for ri, ai in zip(ring_idx, angle_idx):
            hist[ri * n_angles + ai] += 1

        # Normalize
        total = hist.sum()
        if total > 0:
            hist /= total

        desc_img[py, px] = hist

    return desc_img


# ─── Top-Level Normalization ─────────────────────────────────────

def normalize_pair(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    work_dim: int = 840,
    n_clusters: int = 5,
    verbose: bool = True,
) -> NormalizedPair:
    """
    Full preprocessing pipeline:
    1. Segment both images independently
    2. Cross-validate to ensure mutual consistency
    3. Build distortion-tolerant representations
    """
    if verbose:
        print("═" * 60)
        print("  MAP PREPROCESSOR — Land/Sea Segmentation")
        print("═" * 60)

    # Convert to Lab for analysis
    ref_bgr = (
        ref_img if len(ref_img.shape) == 3
        else cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    )
    src_bgr = (
        src_img if len(src_img.shape) == 3
        else cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    )
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2Lab)
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2Lab)

    # 1. Independent segmentation
    if verbose:
        print("\n① Segmenting reference image …")
    ref_seg = segment_land_sea(ref_img, n_clusters, verbose=verbose)

    if verbose:
        print("\n② Segmenting source image …")
    src_seg = segment_land_sea(src_img, n_clusters, verbose=verbose)

    # 2. Cross-validate
    if verbose:
        print("\n③ Cross-validating classifications …")
    ref_seg, src_seg = _cross_validate_segmentation(
        ref_seg, src_seg, ref_lab, src_lab, verbose=verbose
    )

    # 3. Resize to working resolution
    def _resize(img, max_dim):
        h, w = img.shape[:2]
        s = min(max_dim / max(h, w), 1.0)
        nh, nw = max(int(h * s), 32), max(int(w * s), 32)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    ref_land_w = _resize(ref_seg.land_mask, work_dim)
    src_land_w = _resize(src_seg.land_mask, work_dim)
    ref_coast_w = _resize(ref_seg.coastline, work_dim)
    src_coast_w = _resize(src_seg.coastline, work_dim)

    # Re-threshold after resize
    ref_land_w = (ref_land_w > 127).astype(np.uint8) * 255
    src_land_w = (src_land_w > 127).astype(np.uint8) * 255
    ref_coast_w = (ref_coast_w > 64).astype(np.uint8) * 255
    src_coast_w = (src_coast_w > 64).astype(np.uint8) * 255

    # 4. Build representations
    if verbose:
        print("\n④ Building distortion-tolerant representations …")

    ref_dt = _compute_signed_distance(ref_land_w)
    src_dt = _compute_signed_distance(src_land_w)

    ref_coast_dt = cv2.distanceTransform(
        255 - ref_coast_w, cv2.DIST_L2, 5
    ).astype(np.float32)
    src_coast_dt = cv2.distanceTransform(
        255 - src_coast_w, cv2.DIST_L2, 5
    ).astype(np.float32)

    ref_curv = _compute_coastline_curvature(ref_coast_w)
    src_curv = _compute_coastline_curvature(src_coast_w)

    ref_orient = _compute_coastline_orientation(ref_land_w)
    src_orient = _compute_coastline_orientation(src_land_w)

    if verbose:
        print("  Building shape context descriptors …")
    ref_sc = _build_shape_context_image(
        ref_land_w, ref_coast_w
    )
    src_sc = _build_shape_context_image(
        src_land_w, src_coast_w
    )

    if verbose:
        print("\n  ✓ Preprocessing complete")
        print(
            f"    ref work size: "
            f"{ref_land_w.shape[1]}×{ref_land_w.shape[0]}"
        )
        print(
            f"    src work size: "
            f"{src_land_w.shape[1]}×{src_land_w.shape[0]}"
        )

    return NormalizedPair(
        ref_land_mask=ref_land_w,
        src_land_mask=src_land_w,
        ref_coastline=ref_coast_w,
        src_coastline=src_coast_w,
        ref_dt_signed=ref_dt,
        src_dt_signed=src_dt,
        ref_coastline_dt=ref_coast_dt,
        src_coastline_dt=src_coast_dt,
        ref_curvature=ref_curv,
        src_curvature=src_curv,
        ref_orientation=ref_orient,
        src_orientation=src_orient,
        ref_shape_context=ref_sc,
        src_shape_context=src_sc,
        ref_seg=ref_seg,
        src_seg=src_seg,
    )


# ─── Visualization Helpers ───────────────────────────────────────

def visualize_segmentation(
    img: np.ndarray,
    seg: SegmentationResult,
    title: str = "",
    figsize: Tuple[int, int] = (20, 5),
):
    """Show original, land mask, sea mask, coastline side by side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    show = (
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3
        else img
    )
    axes[0].imshow(show, cmap="gray")
    axes[0].set_title(f"{title} Original")
    axes[0].axis("off")

    axes[1].imshow(seg.land_mask, cmap="gray")
    axes[1].set_title(
        f"Land ({(seg.land_mask > 127).mean()*100:.1f}%)"
    )
    axes[1].axis("off")

    axes[2].imshow(seg.sea_mask, cmap="gray")
    axes[2].set_title(
        f"Sea ({(seg.sea_mask > 127).mean()*100:.1f}%)"
    )
    axes[2].axis("off")

    axes[3].imshow(seg.coastline, cmap="hot")
    axes[3].set_title("Coastline")
    axes[3].axis("off")

    plt.suptitle(
        f"Segmentation confidence: {seg.confidence:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def visualize_normalized_pair(
    pair: NormalizedPair,
    figsize: Tuple[int, int] = (24, 12),
):
    """Visualize all representation channels for both images."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=figsize)

    titles = ["Land Mask", "Signed DT", "Curvature", "Orientation"]
    ref_imgs = [
        pair.ref_land_mask,
        pair.ref_dt_signed,
        pair.ref_curvature,
        pair.ref_orientation,
    ]
    src_imgs = [
        pair.src_land_mask,
        pair.src_dt_signed,
        pair.src_curvature,
        pair.src_orientation,
    ]
    cmaps = ["gray", "RdBu_r", "hot", "hsv"]

    for i in range(4):
        axes[0, i].imshow(ref_imgs[i], cmap=cmaps[i])
        axes[0, i].set_title(f"Ref: {titles[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(src_imgs[i], cmap=cmaps[i])
        axes[1, i].set_title(f"Src: {titles[i]}")
        axes[1, i].axis("off")

    # Bottom row: coastline DT and shape context first channel
    axes[2, 0].imshow(pair.ref_coastline_dt, cmap="magma")
    axes[2, 0].set_title("Ref: Coast DT")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(pair.src_coastline_dt, cmap="magma")
    axes[2, 1].set_title("Src: Coast DT")
    axes[2, 1].axis("off")

    if pair.ref_shape_context.shape[2] > 0:
        axes[2, 2].imshow(
            pair.ref_shape_context[:, :, 0], cmap="viridis"
        )
        axes[2, 2].set_title("Ref: Shape Context [0]")
    axes[2, 2].axis("off")

    if pair.src_shape_context.shape[2] > 0:
        axes[2, 3].imshow(
            pair.src_shape_context[:, :, 0], cmap="viridis"
        )
        axes[2, 3].set_title("Src: Shape Context [0]")
    axes[2, 3].axis("off")

    plt.tight_layout()
    plt.show()