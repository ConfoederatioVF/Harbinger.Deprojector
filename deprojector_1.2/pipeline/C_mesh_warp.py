"""
Pipeline Step C: Point-Constrained Mesh Warp (v3 — Affine Triangles & TPS)
Optimizes a sparse set of control points using PyTorch SGD. Supports both 
Affine Triangulation (barycentric rendering) and Thin Plate Spline (TPS) modes.
Points are dynamically added to high-error regions and pruned if redundant during segmentation.
"""

from typing import Tuple, Optional, Any, Callable, List

import cv2
import math
import numpy as np
import scipy.spatial
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
# 1. POLYGON & BBOX UTILITIES
# ─────────────────────────────────────────────────────────────────────

def polygon_tight_bbox(polygon: np.ndarray, ref_shape: Tuple[int, int]) -> dict:
    rh, rw = ref_shape[:2]
    x0 = int(np.floor(polygon[:, 0].min()))
    y0 = int(np.floor(polygon[:, 1].min()))
    x1 = int(np.ceil(polygon[:, 0].max()))
    y1 = int(np.ceil(polygon[:, 1].max()))
    return {"x0": max(0, x0), "y0": max(0, y0), "x1": min(rw - 1, x1), "y1": min(rh - 1, y1)}

def rasterise_polygon_mask(
    polygon: np.ndarray, h: int, w: int, offset_xy: Optional[Tuple[int, int]] = None,
    scale_xy: Optional[Tuple[float, float]] = None
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
    polygon: np.ndarray, h: int, w: int, offset_xy: Optional[Tuple[int, int]] = None,
    scale_xy: Optional[Tuple[float, float]] = None, device: Optional[torch.device] = None
) -> torch.Tensor:
    mask_np = rasterise_polygon_mask(polygon, h, w, offset_xy, scale_xy)
    t = (torch.from_numpy(mask_np).float() / 255.0).unsqueeze(0).unsqueeze(0)
    return t.to(device) if device is not None else t

# ─────────────────────────────────────────────────────────────────────
# 2. CONTROL POINT INITIALIZATION & DYNAMIC MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

def init_control_points(
    bbox: dict, poly_mask_np: np.ndarray, grid_size: int = 4
) -> np.ndarray:
    """Creates an initial coarse grid of points inside the polygon, plus the bounding box corners."""
    h, w = poly_mask_np.shape
    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    
    xs = np.linspace(0, w - 1, grid_size)
    ys = np.linspace(0, h - 1, grid_size)
    xv, yv = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Keep points inside polygon
    valid_pts = []
    for pt in grid_pts:
        if poly_mask_np[int(pt[1]), int(pt[0])] > 0:
            valid_pts.append(pt)
            
    # Always include the 4 corners of the local working bounding box to prevent Delaunay out-of-bounds
    corners = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    valid_pts.extend(corners)
    
    return np.unique(np.array(valid_pts, dtype=np.float32), axis=0)

def apply_affine_to_points(pts: np.ndarray, src_pts: np.ndarray, ref_pts: np.ndarray, bbox: dict) -> np.ndarray:
    """Maps reference points to initial source locations using RANSAC Affine."""
    full_pts = pts.copy()
    full_pts[:, 0] += bbox["x0"]
    full_pts[:, 1] += bbox["y0"]
    
    M, _ = cv2.estimateAffine2D(ref_pts.astype(np.float32), src_pts.astype(np.float32), method=cv2.RANSAC)
    if M is None:
        return pts.copy().astype(np.float32)
        
    ones = np.ones((len(full_pts), 1))
    pts_h = np.hstack([full_pts, ones])
    mapped = (M @ pts_h.T).T
    return mapped.astype(np.float32)

def add_dynamic_points(
    error_map_np: np.ndarray, P_ref_np: np.ndarray, P_src_np: np.ndarray, 
    current_grid_np: np.ndarray, n_points: int = 5, min_dist: int = 20,
    error_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds high-error regions and inserts new control points there."""
    err = cv2.GaussianBlur(error_map_np.astype(np.float32), (9, 9), 0)
    new_refs, new_srcs = [], []
    
    for _ in range(n_points):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(err)
        
        if max_val < error_threshold: 
            break
            
        dist = np.linalg.norm(P_ref_np - np.array(max_loc), axis=1)
        if np.any(dist < min_dist):
            cv2.circle(err, max_loc, min_dist, 0, -1)
            continue
            
        new_refs.append([max_loc[0], max_loc[1]])
        new_srcs.append(current_grid_np[max_loc[1], max_loc[0]]) 
        cv2.circle(err, max_loc, min_dist, 0, -1)
        
    new_refs_np = np.array(new_refs, dtype=np.float32) if new_refs else np.empty((0, 2), dtype=np.float32)
    if new_refs:
        P_ref_np = np.vstack([P_ref_np, new_refs]).astype(np.float32)
        P_src_np = np.vstack([P_src_np, new_srcs]).astype(np.float32)
        
    return P_ref_np, P_src_np, new_refs_np

def prune_control_points(
    P_ref_np: np.ndarray, P_src_np: np.ndarray, warp_mode: str, 
    work_h: int, work_w: int, h_s: int, w_s: int, 
    src_mask_t: torch.Tensor, ref_mask_sm: torch.Tensor, poly_mask_t: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Greedily attempts to remove points if doing so increases IoU."""
    
    def calc_iou(P_r, P_s):
        with torch.no_grad():
            if warp_mode == "affine":
                try:
                    idx, wt, _ = precompute_affine_barycentric(P_r, work_h, work_w)
                    grid = evaluate_affine(torch.tensor(P_s, dtype=torch.float32, device=device), idx.to(device), wt.to(device))
                except Exception:
                    return -1.0 # Failed Delaunay
            else:
                L_inv, U, P_grid = precompute_tps_matrices(torch.tensor(P_r, dtype=torch.float32, device=device), work_h, work_w, device)
                grid = evaluate_tps(torch.tensor(P_s, dtype=torch.float32, device=device), L_inv, U, P_grid, work_h, work_w)
                
            norm_grid_x = (grid[..., 0] / (w_s - 1)) * 2.0 - 1.0
            norm_grid_y = (grid[..., 1] / (h_s - 1)) * 2.0 - 1.0
            norm_grid = torch.stack([norm_grid_x, norm_grid_y], dim=-1).unsqueeze(0)
            
            # Constrain evaluated pixels from exceeding source bounds so we don't sample black padded space
            norm_grid = torch.clamp(norm_grid, -1.0, 1.0)
            
            warped = F.grid_sample(src_mask_t, norm_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            
            pm = poly_mask_t > 0.5
            w_hit = (warped > 0.5) & pm
            r_hit = (ref_mask_sm > 0.5) & pm
            
            intersection = (w_hit & r_hit).sum().float()
            union = (w_hit | r_hit).sum().float().clamp(min=1e-8)
            return (intersection / union).item()

    base_iou = calc_iou(P_ref_np, P_src_np)
    keep_mask = np.ones(len(P_ref_np), dtype=bool)
    removed_count = 0
    
    # Identify corners to protect them from pruning (required for bounds stability)
    is_corner = np.zeros(len(P_ref_np), dtype=bool)
    for i, pt in enumerate(P_ref_np):
        x, y = pt[0], pt[1]
        if (x <= 1 or x >= work_w - 2) and (y <= 1 or y >= work_h - 2):
            is_corner[i] = True
            
    for i in range(len(P_ref_np)):
        if is_corner[i]:
            continue
            
        keep_mask[i] = False
        P_r_test = P_ref_np[keep_mask]
        P_s_test = P_src_np[keep_mask]
        
        new_iou = calc_iou(P_r_test, P_s_test)
        
        if new_iou > base_iou + 1e-5:
            base_iou = new_iou
            removed_count += 1
        else:
            keep_mask[i] = True

    return P_ref_np[keep_mask], P_src_np[keep_mask], removed_count, base_iou

# ─────────────────────────────────────────────────────────────────────
# 3. DIFFERENTIABLE WARPING KERNELS (AFFINE & TPS)
# ─────────────────────────────────────────────────────────────────────

def precompute_affine_barycentric(P_ref_np: np.ndarray, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    delaunay = scipy.spatial.Delaunay(P_ref_np)
    y, x = np.mgrid[0:H, 0:W]
    grid_pts = np.column_stack([x.ravel(), y.ravel()])
    
    simplices = delaunay.find_simplex(grid_pts)
    
    T_inv = delaunay.transform[simplices, :2]
    C = delaunay.transform[simplices, 2]
    B = grid_pts - C
    
    u = T_inv[:, 0, 0] * B[:, 0] + T_inv[:, 0, 1] * B[:, 1]
    v = T_inv[:, 1, 0] * B[:, 0] + T_inv[:, 1, 1] * B[:, 1]
    w = 1.0 - u - v
    
    weights = np.column_stack([u, v, w])
    indices = delaunay.simplices[simplices]
    
    idx_t = torch.from_numpy(indices).long().reshape(H, W, 3)
    wt_t = torch.from_numpy(weights).float().reshape(H, W, 3)
    return idx_t, wt_t, delaunay.simplices

def evaluate_affine(P_src: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    src_triangles = P_src[indices] 
    return (src_triangles * weights.unsqueeze(-1)).sum(dim=2)

def precompute_tps_matrices(P_ref: torch.Tensor, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    P_ref = P_ref.float()
    N = P_ref.shape[0]
    dist = torch.cdist(P_ref, P_ref)
    K = dist ** 2 * torch.log(dist + 1e-8)
    P = torch.cat([torch.ones(N, 1, device=device), P_ref], dim=1)
    
    L = torch.zeros(N + 3, N + 3, device=device)
    L[:N, :N] = K
    L[:N, N:] = P
    L[N:, :N] = P.t()
    L = L + torch.eye(N + 3, device=device) * 1e-5 
    
    L_inv = torch.linalg.pinv(L)
    
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_pts = torch.stack([x.flatten(), y.flatten()], dim=1).float()
    
    dist_grid = torch.cdist(grid_pts, P_ref)
    U = dist_grid ** 2 * torch.log(dist_grid + 1e-8)
    P_grid = torch.cat([torch.ones(H*W, 1, device=device), grid_pts], dim=1)
    
    return L_inv, U, P_grid

def evaluate_tps(P_src: torch.Tensor, L_inv: torch.Tensor, U: torch.Tensor, P_grid: torch.Tensor, H: int, W: int) -> torch.Tensor:
    N = U.shape[1]
    Y = torch.cat([P_src, torch.zeros(3, 2, device=P_src.device)], dim=0)
    coeffs = L_inv @ Y
    W_mat = coeffs[:N, :]
    A_mat = coeffs[N:, :]
    
    grid_out = U @ W_mat + P_grid @ A_mat
    return grid_out.view(H, W, 2)

# ─────────────────────────────────────────────────────────────────────
# 4. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def fold_loss_affine(P_src: torch.Tensor, simplices: np.ndarray) -> torch.Tensor:
    A = P_src[simplices[:, 0]]
    B = P_src[simplices[:, 1]]
    C = P_src[simplices[:, 2]]
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    return F.relu(-area + 0.1).mean()

def bend_loss_tps(P_src: torch.Tensor, L_inv: torch.Tensor, N: int) -> torch.Tensor:
    Y = torch.cat([P_src, torch.zeros(3, 2, device=P_src.device)], dim=0)
    coeffs = L_inv @ Y
    W_mat = coeffs[:N, :]
    return (W_mat ** 2).mean()

def dice_loss_poly(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    poly_mask: torch.Tensor, 
    weight_map: Optional[torch.Tensor] = None, 
    smooth: float = 1.0
) -> torch.Tensor:
    w = poly_mask
    if weight_map is not None:
        w = w * weight_map
        
    p = pred.reshape(-1)
    t = target.reshape(-1)
    w_flat = w.reshape(-1)
    
    inter = (w_flat * p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / ((w_flat * p).sum() + (w_flat * t).sum() + smooth)

def mse_loss_poly(
    pred: torch.Tensor, target: torch.Tensor, poly_mask: torch.Tensor, 
    weight_map: Optional[torch.Tensor] = None
) -> torch.Tensor:
    n = poly_mask.sum().clamp(min=1.0)
    diff_sq = (pred - target) ** 2 * poly_mask
    if weight_map is not None:
        diff_sq = diff_sq * weight_map
    return diff_sq.sum() / n

# ─────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def show_extent_initialisation(ref_img: np.ndarray, src_img: np.ndarray, extent_result: ExtentResult, bbox: dict):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB) if len(src_img.shape) == 3 else src_img
    axes[0].imshow(src_rgb, cmap="gray")
    axes[0].set_title(f"Source  {src_img.shape[1]}×{src_img.shape[0]}")
    axes[0].axis("off")

    vis = ref_img.copy()
    if extent_result.polygon is not None:
        pts = extent_result.polygon.astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 200, 255), 2)

    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Extent polygon (green) + tight bbox (cyan)\nconf={extent_result.confidence:.3f}")
    axes[1].axis("off")

    vis2 = ref_img.copy()
    if extent_result.ref_pts is not None and len(extent_result.ref_pts) > 0:
        n_show = min(200, len(extent_result.ref_pts))
        idx = np.random.choice(len(extent_result.ref_pts), n_show, replace=False)
        for i in idx:
            cv2.circle(vis2, (int(extent_result.ref_pts[i, 0]), int(extent_result.ref_pts[i, 1])), 4, (0, 0, 255), -1)
        if extent_result.polygon is not None:
            cv2.polylines(vis2, [extent_result.polygon.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    
    axes[2].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Inlier correspondences")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

def show_mask_comparison_poly(warped_mask: torch.Tensor, ref_mask: torch.Tensor, poly_mask: torch.Tensor, title: str = ""):
    wm = warped_mask[0, 0].cpu().numpy()
    rm = ref_mask[0, 0].cpu().numpy()
    pm = poly_mask[0, 0].cpu().numpy() > 0.5

    fig, axes = plt.subplots(1, 4, figsize=(28, 5))
    axes[0].imshow(wm, cmap="gray"); axes[0].set_title("Warped Source Mask")
    axes[1].imshow(rm, cmap="gray"); axes[1].set_title("Reference Mask (bbox crop)")
    axes[2].imshow(pm, cmap="gray"); axes[2].set_title("Polygon Mask")

    rgb = np.zeros((*rm.shape, 3))
    hit, miss, extra = (wm > 0.5) & (rm > 0.5) & pm, (rm > 0.5) & (wm <= 0.5) & pm, (wm > 0.5) & (rm <= 0.5) & pm
    rgb[hit, 1] = 1.0; rgb[miss, 0] = 1.0; rgb[extra, 2] = 1.0
    axes[3].imshow(rgb)
    axes[3].set_title("Within polygon: G=hit R=miss B=extra")

    for ax in axes: ax.axis("off")
    if title: fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_points_mesh(P_ref_np: np.ndarray, ref_img_np: np.ndarray, warp_mode: str, simplices: Optional[np.ndarray] = None, title: str = "Mesh"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(ref_img_np, cv2.COLOR_BGR2RGB))
    
    if warp_mode == "affine" and simplices is not None:
        ax.triplot(P_ref_np[:, 0], P_ref_np[:, 1], simplices, color='cyan', lw=1.2, alpha=0.8)
    
    ax.plot(P_ref_np[:, 0], P_ref_np[:, 1], 'yo', markersize=5, markeredgecolor='black')
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def show_displacement_quiver(P_init: np.ndarray, P_final: np.ndarray, title: str = "Control Point Movement"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(P_init[:, 0], P_init[:, 1], c='blue', s=20, label='Initial Position', alpha=0.6)
    ax.scatter(P_final[:, 0], P_final[:, 1], c='red', s=20, label='Optimized Position', alpha=0.6)
    
    for i in range(len(P_init)):
        ax.arrow(P_init[i, 0], P_init[i, 1], P_final[i, 0] - P_init[i, 0], P_final[i, 1] - P_init[i, 1],
                 head_width=2, head_length=3, fc='black', ec='black', alpha=0.4)
        
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis() 
    plt.show()

# ─────────────────────────────────────────────────────────────────────
# 6. MAIN SGD POINT OPTIMIZATION PIPELINE
# ─────────────────────────────────────────────────────────────────────

def sgd_point_warp_polygon_constrained(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    extent_result: ExtentResult,
    work_h_bbox: int = 384,
    warp_mode: str = "affine",
    levels: int = 4,
    steps_per_lvl: int = 350,
    lr_gain: float = 0.6,
    lr_init: float = 2.0,
    lam_fold: float = 0.5,
    dyn_points_per_level: int = 8,
    points_gain: float = 1,
    dyn_error_threshold: float = 0.05,
    dyn_min_dist: int = 15,
    prune_interval: int = 150,
    edge_penalty: float = 0.0,
    center_penalty: float = 0.0,
    edge_threshold: int = 8,
    edge_weight: float = 1.0,
    fill_weight: float = 1.0,
    show_plots: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict, float]:
    
    if extent_result.polygon is None:
        raise ValueError("extent_result.polygon is None — run find_extent() first.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {warp_mode.upper()}")
    
    h_r, w_r = ref_img.shape[:2]
    h_s, w_s = src_img.shape[:2]
    
    # ── Phase 1: Context & Bounding Box ───────────────────────────
    polygon = extent_result.polygon
    bbox = polygon_tight_bbox(polygon, ref_img.shape)
    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    bw, bh = bx1 - bx0, by1 - by0
    
    if show_plots:
        show_extent_initialisation(ref_img, src_img, extent_result, bbox)
    
    work_w_bbox = int(work_h_bbox * (bw / bh))
    scale_x, scale_y = work_w_bbox / bw, work_h_bbox / bh
    
    # ── Phase 2: Masks, Tensors & Edge Weights ────────────────────
    mask_ref_full = extract_land_mask(ref_img)
    mask_src_full = extract_land_mask(src_img)
    
    poly_mask_work_np = rasterise_polygon_mask(polygon, work_h_bbox, work_w_bbox, offset_xy=(bx0, by0), scale_xy=(scale_x, scale_y))
    poly_mask_t = (torch.from_numpy(poly_mask_work_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    ref_bbox_t = (torch.from_numpy(mask_ref_full[by0:by1, bx0:bx1]).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    ref_bbox_sm = F.interpolate(ref_bbox_t, (work_h_bbox, work_w_bbox), mode="bilinear", align_corners=True)
    
    src_mask_t = (torch.from_numpy(mask_src_full).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    src_rgb_np = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img_t = (torch.from_numpy(src_rgb_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)

    # Compute Spatial Weight Map for MSE if requested (edge and/or center penalties via extents)
    spatial_weight_map = None
    if edge_penalty > 0.0 or center_penalty > 0.0:
        y_grid, x_grid = torch.meshgrid(torch.arange(work_h_bbox, device=device), torch.arange(work_w_bbox, device=device), indexing='ij')
        # Normalize to [-1, 1] for both X and Y
        nx = (x_grid / (work_w_bbox - 1)) * 2.0 - 1.0
        ny = (y_grid / (work_h_bbox - 1)) * 2.0 - 1.0
        # Chebyshev distance from the center (0 in center, 1 at edge)
        dist_from_center = torch.max(torch.abs(nx), torch.abs(ny))
        
        # Linearly scale penalty:
        e_weight = 1.0 + (edge_penalty * dist_from_center) + (center_penalty * (1.0 - dist_from_center))
        spatial_weight_map = e_weight.unsqueeze(0).unsqueeze(0).float()
        
    # Generate Custom Binary Weight Map for Jaccard (Dice) 
    poly_jaccard_weight_np = np.zeros_like(poly_mask_work_np, dtype=np.float32)
    # Strictly binary map >127 mapping to 0 or 255 purely to avoid antialiasing 
    binary_poly = (poly_mask_work_np > 127).astype(np.uint8) * 255 
    
    if edge_threshold > 0:
        # Distance transform computes exact depth of polygon pixels to the edge, completely avoiding gradient aliasing
        dist_transform = cv2.distanceTransform(binary_poly, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        poly_jaccard_weight_np[dist_transform > edge_threshold] = fill_weight
        poly_jaccard_weight_np[(dist_transform > 0) & (dist_transform <= edge_threshold)] = edge_weight
    else:
        poly_jaccard_weight_np[binary_poly > 0] = fill_weight
        
    jaccard_weight_map_t = torch.from_numpy(poly_jaccard_weight_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # ── Phase 3: Control Point Initialization ─────────────────────
    print("\n▸ Phase 3: Initializing Control Points")
    P_ref_np = init_control_points(bbox, poly_mask_work_np, grid_size=6)
    P_src_np = apply_affine_to_points(P_ref_np / [scale_x, scale_y], extent_result.src_pts, extent_result.ref_pts, bbox)
    P_src_original_init = P_src_np.copy()
    
    print(f"  Initialized {len(P_ref_np)} control points.")
    
    # ── Phase 4: Dynamic SGD Optimization Loop ────────────────────
    print(f"\n▸ Phase 4: Multi-Level {warp_mode.upper()} SGD Point Optimization")
    
    current_simplices = None
    
    for lvl in range(levels):
        lr = lr_init * (lr_gain ** lvl)
        print(f"  ── Level {lvl + 1}/{levels} | Points: {len(P_ref_np)} | LR: {lr:.4f}")
        
        if warp_mode == "affine":
            idx_t, wt_t, current_simplices = precompute_affine_barycentric(P_ref_np, work_h_bbox, work_w_bbox)
            idx_t, wt_t = idx_t.to(device), wt_t.to(device)
        else:
            L_inv, U, P_grid = precompute_tps_matrices(torch.from_numpy(P_ref_np).float().to(device), work_h_bbox, work_w_bbox, device)
            
        P_src_t = torch.tensor(P_src_np, dtype=torch.float32, device=device, requires_grad=True)
        optim = torch.optim.Adam([P_src_t], lr=lr)
        
        best_loss = float('inf')
        best_P_src = P_src_np.copy()
        losses_log = []
        
        # SGD Loop
        for step in range(steps_per_lvl):
            optim.zero_grad()
            
            if warp_mode == "affine":
                grid_src = evaluate_affine(P_src_t, idx_t, wt_t)
            else:
                grid_src = evaluate_tps(P_src_t, L_inv, U, P_grid, work_h_bbox, work_w_bbox)
                
            norm_grid_x = (grid_src[..., 0] / (w_s - 1)) * 2.0 - 1.0
            norm_grid_y = (grid_src[..., 1] / (h_s - 1)) * 2.0 - 1.0
            norm_grid = torch.stack([norm_grid_x, norm_grid_y], dim=-1).unsqueeze(0)
            
            # Constrain pixels from exceeding the source image bounds
            # This prevents clipping to black, while allowing actual control points to overshoot.
            norm_grid = torch.clamp(norm_grid, -1.0, 1.0)
            
            warped = F.grid_sample(src_mask_t, norm_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            
            l_dice = dice_loss_poly(warped, ref_bbox_sm, poly_mask_t, weight_map=jaccard_weight_map_t)
            l_mse = mse_loss_poly(warped, ref_bbox_sm, poly_mask_t, weight_map=spatial_weight_map)
            
            if warp_mode == "affine":
                l_struct = fold_loss_affine(P_src_t, current_simplices)
            else:
                l_struct = bend_loss_tps(P_src_t, L_inv, len(P_ref_np))
                
            loss = l_dice + l_mse + lam_fold * l_struct
            loss.backward()
            optim.step()
            
            losses_log.append(loss.item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_P_src = P_src_t.detach().cpu().numpy()
                
            if (step + 1) % 100 == 0 or step == steps_per_lvl - 1:
                print(f"     step {step+1:4d} | loss={loss.item():.4f} | dice={l_dice.item():.4f} | struct={l_struct.item():.4f}")

            # ── Prune Redundant Control Points Check ────────────────
            if prune_interval > 0 and (step + 1) % prune_interval == 0 and (step + 1) < steps_per_lvl:
                curr_P_src = P_src_t.detach().cpu().numpy()
                new_P_ref, new_P_src, removed_n, new_iou = prune_control_points(
                    P_ref_np, curr_P_src, warp_mode, work_h_bbox, work_w_bbox, 
                    h_s, w_s, src_mask_t, ref_bbox_sm, poly_mask_t, device
                )
                
                if removed_n > 0:
                    print(f"     [Pruning] Step {step+1}: Successfully removed {removed_n} point(s) improving IoU to {new_iou:.4f}")
                    P_ref_np, P_src_np = new_P_ref, new_P_src
                    
                    if warp_mode == "affine":
                        idx_t, wt_t, current_simplices = precompute_affine_barycentric(P_ref_np, work_h_bbox, work_w_bbox)
                        idx_t, wt_t = idx_t.to(device), wt_t.to(device)
                    else:
                        L_inv, U, P_grid = precompute_tps_matrices(torch.from_numpy(P_ref_np).float().to(device), work_h_bbox, work_w_bbox, device)
                        
                    P_src_t = torch.tensor(P_src_np, dtype=torch.float32, device=device, requires_grad=True)
                    optim = torch.optim.Adam([P_src_t], lr=lr)
                    
                    best_loss = float('inf')
                    best_P_src = P_src_np.copy()


        # Post-level updates
        P_src_np = best_P_src.copy()
        
        if show_plots:
            plt.figure(figsize=(6, 3))
            plt.plot(losses_log, linewidth=1.5)
            plt.title(f"Level {lvl + 1} Loss Curve")
            plt.grid(True, alpha=0.3)
            plt.show()

            with torch.no_grad():
                if warp_mode == "affine":
                    final_grid = evaluate_affine(torch.from_numpy(P_src_np).to(device), idx_t, wt_t)
                else:
                    final_grid = evaluate_tps(torch.from_numpy(P_src_np).to(device), L_inv, U, P_grid, work_h_bbox, work_w_bbox)
                n_grid = torch.stack([(final_grid[..., 0]/(w_s-1))*2-1, (final_grid[..., 1]/(h_s-1))*2-1], dim=-1).unsqueeze(0)
                
                # Constrain pixels from exceeding the source image bounds
                n_grid = torch.clamp(n_grid, -1.0, 1.0)
                
                w_mask = F.grid_sample(src_mask_t, n_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            show_mask_comparison_poly(w_mask, ref_bbox_sm, poly_mask_t, title=f"End of Level {lvl+1}")

        # Add Dynamic Points if not the last level
        if lvl < levels - 1:
            with torch.no_grad():
                # Apply the spatial penalty weight to the error map so dynamic points spawn at penalized regions more aggressively
                base_err = torch.abs(w_mask - ref_bbox_sm) * poly_mask_t
                if spatial_weight_map is not None:
                    base_err = base_err * spatial_weight_map
                error_map = base_err[0, 0].cpu().numpy()
                
            old_len = len(P_ref_np)
            
            dyn_points_per_level = math.ceil(dyn_points_per_level*points_gain)
            
            P_ref_np, P_src_np, new_pts = add_dynamic_points(
                error_map, P_ref_np, P_src_np, final_grid.cpu().numpy(), 
                n_points=dyn_points_per_level, 
                min_dist=dyn_min_dist, 
                error_threshold=dyn_error_threshold
            )
            print(f"     [Dynamic] Added {len(P_ref_np) - old_len} points in high-error regions.")

    if show_plots:
        ref_crop_np = ref_img[by0:by1, bx0:bx1]
        ref_crop_rs = cv2.resize(ref_crop_np, (work_w_bbox, work_h_bbox))
        show_points_mesh(P_ref_np, ref_crop_rs, warp_mode, current_simplices, f"Optimized Mesh ({warp_mode.upper()})")
        show_displacement_quiver(P_src_original_init, P_src_np[:len(P_src_original_init)], "Control Point Displacements (Src Image Coordinates)")

    # ── Phase 5: Full-Resolution Application ──────────────────────
    print("\n▸ Phase 5: Rendering Full-Resolution Result")
    
    P_ref_full = P_ref_np.copy().astype(np.float32)
    P_ref_full[:, 0] /= scale_x
    P_ref_full[:, 1] /= scale_y
    
    if warp_mode == "affine":
        idx_f, wt_f, _ = precompute_affine_barycentric(P_ref_full, bh, bw)
        grid_full = evaluate_affine(torch.from_numpy(P_src_np).to(device), idx_f.to(device), wt_f.to(device))
    else:
        L_inv_f, U_f, P_grid_f = precompute_tps_matrices(torch.from_numpy(P_ref_full).float().to(device), bh, bw, device)
        grid_full = evaluate_tps(torch.from_numpy(P_src_np).to(device), L_inv_f, U_f, P_grid_f, bh, bw)

    norm_gx = (grid_full[..., 0] / (w_s - 1)) * 2.0 - 1.0
    norm_gy = (grid_full[..., 1] / (h_s - 1)) * 2.0 - 1.0
    norm_grid_full = torch.stack([norm_gx, norm_gy], dim=-1).unsqueeze(0)
    
    # Constrain pixels from exceeding the source image bounds
    norm_grid_full = torch.clamp(norm_grid_full, -1.0, 1.0)
    
    poly_full_np = rasterise_polygon_mask(polygon, bh, bw, offset_xy=(bx0, by0))
    poly_full_t = (torch.from_numpy(poly_full_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        warped_rgb = F.grid_sample(src_img_t, norm_grid_full, mode="bilinear", padding_mode="zeros", align_corners=True)
        warped_mask = F.grid_sample(src_mask_t, norm_grid_full, mode="bilinear", padding_mode="zeros", align_corners=True)
        
        warped_rgb = warped_rgb * poly_full_t
        warped_mask = warped_mask * poly_full_t

    # Reconstruct Canvas
    sea_color = get_sea_color(ref_img)
    canvas_np = np.full_like(ref_img, sea_color)
    
    warped_bbox_np = (warped_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    warped_bbox_bgr = cv2.cvtColor(warped_bbox_np, cv2.COLOR_RGB2BGR)
    
    poly_hw = poly_full_np > 127
    region = canvas_np[by0:by1, bx0:bx1]
    region[poly_hw] = warped_bbox_bgr[poly_hw]
    canvas_np[by0:by1, bx0:bx1] = region
    
    # Evaluate Final IoU
    mask_canvas = np.zeros((h_r, w_r), dtype=np.float32)
    mask_canvas[by0:by1, bx0:bx1] = (warped_mask[0, 0].cpu().numpy() * (poly_full_np / 255.0))
    
    w_mask_full_t = torch.from_numpy(mask_canvas).float().unsqueeze(0).unsqueeze(0).to(device)
    r_mask_full_t = (torch.from_numpy(mask_ref_full).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    p_full_t = (torch.from_numpy(rasterise_polygon_mask(polygon, h_r, w_r)).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    pm = p_full_t > 0.5
    w_hit = (w_mask_full_t > 0.5) & pm
    r_hit = (r_mask_full_t > 0.5) & pm
    final_iou = (w_hit & r_hit).sum().float() / (w_hit | r_hit).sum().float().clamp(min=1.0)
    
    print(f"  Final Polygon IoU: {final_iou.item():.4f}")
    
    # ── Phase 6: Result Visualization ─────────────────────────────
    if show_plots:
        show_mask_comparison_poly(w_mask_full_t[:, :, by0:by1, bx0:bx1], 
                                  r_mask_full_t[:, :, by0:by1, bx0:bx1], 
                                  poly_full_t, title=f"Final Full-Res Overlap (IoU={final_iou.item():.4f})")
        
        plt.figure(figsize=(12, 6))
        overlay = cv2.addWeighted(ref_img, 0.45, canvas_np, 0.55, 0)
        poly_mask_vis = rasterise_polygon_mask(polygon, h_r, w_r)
        overlay[poly_mask_vis == 0] = ref_img[poly_mask_vis == 0]
        cv2.polylines(overlay, [polygon.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Overlay Result (IoU: {final_iou.item():.4f} | Mode: {warp_mode.upper()})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        axes[0].imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Source")
        axes[1].imshow(cv2.cvtColor(warped_bbox_bgr, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Warped Source (bbox crop)")
        axes[2].imshow(cv2.cvtColor(ref_img[by0:by1, bx0:bx1], cv2.COLOR_BGR2RGB))
        axes[2].set_title("Reference (bbox crop)")
        for ax in axes: ax.axis("off")
        fig.suptitle(f"Bbox Region Comparison  —  polygon IoU = {final_iou:.4f}", fontsize=15)
        plt.tight_layout()
        plt.show()

    print("\n✓ Done.")
    return canvas_np, grid_full.cpu().numpy(), bbox, final_iou.item()