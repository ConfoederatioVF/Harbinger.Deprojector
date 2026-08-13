"""
Pipeline Step C: Point-Constrained Mesh Warp (v3 — Affine Triangles & TPS)
Optimizes a sparse set of control points using PyTorch SGD. Supports both 
Affine Triangulation (barycentric rendering) and Thin Plate Spline (TPS) modes.

[UPDATE]: Smart Ensemble Hybridizer with Performance-Guided NMS!
Evaluates local IoU/Error for both models and creates a smooth Weight Map.
Selects control points based on local model superiority and blends their mapped 
positions based on the combination of performance and spatial priors.
"""

from typing import Tuple, Optional, Any, Callable, List

import cv2
import numpy as np
import scipy.spatial
import scipy.spatial.distance
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
    h, w = poly_mask_np.shape
    bx0, by0, bx1, by1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    
    xs = np.linspace(0, w - 1, grid_size)
    ys = np.linspace(0, h - 1, grid_size)
    xv, yv = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xv.ravel(), yv.ravel()])
    
    valid_pts = []
    for pt in grid_pts:
        if poly_mask_np[int(pt[1]), int(pt[0])] > 0:
            valid_pts.append(pt)
            
    corners = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    valid_pts.extend(corners)
    
    return np.unique(np.array(valid_pts, dtype=np.float32), axis=0)

def apply_affine_to_points(pts: np.ndarray, src_pts: np.ndarray, ref_pts: np.ndarray, bbox: dict) -> np.ndarray:
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
    err = cv2.GaussianBlur(error_map_np.astype(np.float32), (9, 9), 0)
    new_refs, new_srcs = [], []
    
    for _ in range(n_points):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(err)
        if max_val < error_threshold: break
            
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
    
    def calc_iou(P_r, P_s):
        with torch.no_grad():
            if warp_mode == "affine":
                try:
                    idx, wt, _ = precompute_affine_barycentric(P_r, work_h, work_w)
                    grid = evaluate_affine(torch.tensor(P_s, dtype=torch.float32, device=device), idx.to(device), wt.to(device))
                except Exception:
                    return -1.0 
            else:
                L_inv, U, P_grid = precompute_tps_matrices(torch.tensor(P_r, dtype=torch.float32, device=device), work_h, work_w, device)
                grid = evaluate_tps(torch.tensor(P_s, dtype=torch.float32, device=device), L_inv, U, P_grid, work_h, work_w)
                
            norm_grid_x = (grid[..., 0] / (w_s - 1)) * 2.0 - 1.0
            norm_grid_y = (grid[..., 1] / (h_s - 1)) * 2.0 - 1.0
            norm_grid = torch.stack([norm_grid_x, norm_grid_y], dim=-1).unsqueeze(0)
            
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
    
    is_corner = np.zeros(len(P_ref_np), dtype=bool)
    for i, pt in enumerate(P_ref_np):
        x, y = pt[0], pt[1]
        if (x <= 1 or x >= work_w - 2) and (y <= 1 or y >= work_h - 2):
            is_corner[i] = True
            
    for i in range(len(P_ref_np)):
        if is_corner[i]: continue
            
        keep_mask[i] = False
        new_iou = calc_iou(P_ref_np[keep_mask], P_src_np[keep_mask])
        
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
    return (P_src[indices] * weights.unsqueeze(-1)).sum(dim=2)

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
    return (U @ W_mat + P_grid @ A_mat).view(H, W, 2)

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
    return ((L_inv @ Y)[:N, :] ** 2).mean()

def dice_loss_poly(pred: torch.Tensor, target: torch.Tensor, poly_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p, t = (pred * poly_mask).reshape(-1), (target * poly_mask).reshape(-1)
    return 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

def mse_loss_poly(pred: torch.Tensor, target: torch.Tensor, poly_mask: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    diff_sq = (pred - target) ** 2 * poly_mask
    if weight_map is not None: diff_sq = diff_sq * weight_map
    return diff_sq.sum() / poly_mask.sum().clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def eval_warp_at_points_np(P_query: np.ndarray, P_ref: np.ndarray, P_src: np.ndarray, warp_mode: str) -> np.ndarray:
    if warp_mode == "affine":
        delaunay = scipy.spatial.Delaunay(P_ref)
        simplices = delaunay.find_simplex(P_query)
        valid = simplices != -1
        P_src_query = np.zeros_like(P_query, dtype=np.float32)
        
        if np.any(valid):
            T_inv = delaunay.transform[simplices[valid], :2]
            C = delaunay.transform[simplices[valid], 2]
            B = P_query[valid] - C
            u = T_inv[:, 0, 0] * B[:, 0] + T_inv[:, 0, 1] * B[:, 1]
            v = T_inv[:, 1, 0] * B[:, 0] + T_inv[:, 1, 1] * B[:, 1]
            w = 1.0 - u - v
            weights = np.column_stack([u, v, w])
            indices = delaunay.simplices[simplices[valid]]
            P_src_query[valid] = (P_src[indices] * weights[:, :, None]).sum(axis=1)
            
        if np.any(~valid):
            from scipy.spatial import KDTree
            _, nn_idx = KDTree(P_ref).query(P_query[~valid])
            P_src_query[~valid] = P_src[nn_idx]
            
        return P_src_query
    else:
        dist = scipy.spatial.distance.cdist(P_query, P_ref)
        U = dist ** 2 * np.log(dist + 1e-8)
        dist_ref = scipy.spatial.distance.cdist(P_ref, P_ref)
        K = dist_ref ** 2 * np.log(dist_ref + 1e-8)
        
        N = P_ref.shape[0]
        P = np.hstack([np.ones((N, 1)), P_ref])
        
        L = np.zeros((N + 3, N + 3))
        L[:N, :N] = K
        L[:N, N:] = P
        L[N:, :N] = P.T
        L += np.eye(N + 3) * 1e-5
        
        L_inv = np.linalg.pinv(L)
        coeffs = L_inv @ np.vstack([P_src, np.zeros((3, 2))])
        
        return U @ coeffs[:N, :] + np.hstack([np.ones((len(P_query), 1)), P_query]) @ coeffs[N:, :]


# ─────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION HELPERS
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
# 7. MAIN SGD ENSEMBLE PIPELINE
# ─────────────────────────────────────────────────────────────────────

def sgd_point_warp_polygon_constrained(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    extent_result: ExtentResult,
    work_h_bbox: int = 384,
    warp_mode: str = "affine",
    levels: int = 4,
    steps_per_lvl: int = 350,
    lr_init: float = 2.0,
    lam_fold: float = 0.5,
    dyn_points_per_level: int = 8,
    dyn_error_threshold: float = 0.05,
    dyn_min_dist: int = 15,
    prune_interval: int = 150,
    edge_penalty: float = 2.0, # High edge penalty for Model A
    show_plots: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict, float]:
    
    if extent_result.polygon is None:
        raise ValueError("extent_result.polygon is None — run find_extent() first.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {warp_mode.upper()} | Performance-Guided Ensemble Active")
    
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
    
    # ── Phase 2: Shared Masks ───────────────────────────
    mask_ref_full = extract_land_mask(ref_img)
    mask_src_full = extract_land_mask(src_img)
    
    poly_mask_work_np = rasterise_polygon_mask(polygon, work_h_bbox, work_w_bbox, offset_xy=(bx0, by0), scale_xy=(scale_x, scale_y))
    poly_mask_t = (torch.from_numpy(poly_mask_work_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    ref_bbox_t = (torch.from_numpy(mask_ref_full[by0:by1, bx0:bx1]).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    ref_bbox_sm = F.interpolate(ref_bbox_t, (work_h_bbox, work_w_bbox), mode="bilinear", align_corners=True)
    
    src_mask_t = (torch.from_numpy(mask_src_full).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    src_img_t = (torch.from_numpy(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)

    def get_edge_weight_map(penalty: float) -> Optional[torch.Tensor]:
        if penalty <= 0.0: return None
        y_grid, x_grid = torch.meshgrid(torch.arange(work_h_bbox, device=device), torch.arange(work_w_bbox, device=device), indexing='ij')
        nx = (x_grid / (work_w_bbox - 1)) * 2.0 - 1.0
        ny = (y_grid / (work_h_bbox - 1)) * 2.0 - 1.0
        return (1.0 + penalty * torch.max(torch.abs(nx), torch.abs(ny))).unsqueeze(0).unsqueeze(0).float()

    # ── Phase 3 & 4: Ensemble Optimization Logic ───────
    def run_optimization(penalty_val: float, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"\n▸ Running {model_name} (Edge Penalty: {penalty_val:.2f})")
        edge_weight_map = get_edge_weight_map(penalty_val)
        
        P_ref_np = init_control_points(bbox, poly_mask_work_np, grid_size=6)
        P_src_np = apply_affine_to_points(P_ref_np / [scale_x, scale_y], extent_result.src_pts, extent_result.ref_pts, bbox)
        
        # Save initial for quiver plot
        P_src_original_init = P_src_np.copy()
        current_simplices = None
        
        for lvl in range(levels):
            lr = lr_init * (0.6 ** lvl)
            print(f"  ── Level {lvl + 1}/{levels} | Points: {len(P_ref_np)} | LR: {lr:.4f}")
            
            if warp_mode == "affine":
                idx_t, wt_t, current_simplices = precompute_affine_barycentric(P_ref_np, work_h_bbox, work_w_bbox)
                idx_t, wt_t = idx_t.to(device), wt_t.to(device)
            else:
                L_inv, U, P_grid = precompute_tps_matrices(torch.from_numpy(P_ref_np).float().to(device), work_h_bbox, work_w_bbox, device)
                
            P_src_t = torch.tensor(P_src_np, dtype=torch.float32, device=device, requires_grad=True)
            optim = torch.optim.Adam([P_src_t], lr=lr)
            
            best_loss, best_P_src = float('inf'), P_src_np.copy()
            losses_log = []
            
            for step in range(steps_per_lvl):
                optim.zero_grad()
                
                if warp_mode == "affine":
                    grid_src = evaluate_affine(P_src_t, idx_t, wt_t)
                else:
                    grid_src = evaluate_tps(P_src_t, L_inv, U, P_grid, work_h_bbox, work_w_bbox)
                    
                n_gx = (grid_src[..., 0] / (w_s - 1)) * 2.0 - 1.0
                n_gy = (grid_src[..., 1] / (h_s - 1)) * 2.0 - 1.0
                warped = F.grid_sample(src_mask_t, torch.stack([n_gx, n_gy], dim=-1).unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=True)
                
                l_dice = dice_loss_poly(warped, ref_bbox_sm, poly_mask_t)
                l_mse = mse_loss_poly(warped, ref_bbox_sm, poly_mask_t, edge_weight_map)
                l_struct = fold_loss_affine(P_src_t, current_simplices) if warp_mode == "affine" else bend_loss_tps(P_src_t, L_inv, len(P_ref_np))
                
                loss = l_dice + l_mse + lam_fold * l_struct
                loss.backward()
                optim.step()
                
                losses_log.append(loss.item())
                
                if loss.item() < best_loss: 
                    best_loss, best_P_src = loss.item(), P_src_t.detach().cpu().numpy()
                    
                if (step + 1) % 100 == 0 or step == steps_per_lvl - 1:
                    print(f"     step {step+1:4d} | loss={loss.item():.4f} | dice={l_dice.item():.4f} | struct={l_struct.item():.4f}")
                
                if prune_interval > 0 and (step + 1) % prune_interval == 0 and (step + 1) < steps_per_lvl:
                    new_ref, new_src, removed_n, new_iou = prune_control_points(P_ref_np, P_src_t.detach().cpu().numpy(), warp_mode, work_h_bbox, work_w_bbox, h_s, w_s, src_mask_t, ref_bbox_sm, poly_mask_t, device)
                    if removed_n > 0:
                        print(f"     [Pruning] Step {step+1}: Successfully removed {removed_n} point(s) improving IoU to {new_iou:.4f}")
                        P_ref_np, P_src_np = new_ref, new_src
                        if warp_mode == "affine":
                            idx_t, wt_t, current_simplices = precompute_affine_barycentric(P_ref_np, work_h_bbox, work_w_bbox)
                            idx_t, wt_t = idx_t.to(device), wt_t.to(device)
                        else:
                            L_inv, U, P_grid = precompute_tps_matrices(torch.from_numpy(P_ref_np).float().to(device), work_h_bbox, work_w_bbox, device)
                        P_src_t = torch.tensor(P_src_np, dtype=torch.float32, device=device, requires_grad=True)
                        optim = torch.optim.Adam([P_src_t], lr=lr)
                        
                        best_loss = float('inf')
                        best_P_src = P_src_np.copy()

            P_src_np = best_P_src.copy()
            
            with torch.no_grad():
                if warp_mode == "affine": final_grid = evaluate_affine(torch.from_numpy(P_src_np).to(device), idx_t, wt_t)
                else: final_grid = evaluate_tps(torch.from_numpy(P_src_np).to(device), L_inv, U, P_grid, work_h_bbox, work_w_bbox)
                n_grid = torch.stack([(final_grid[..., 0]/(w_s-1))*2-1, (final_grid[..., 1]/(h_s-1))*2-1], dim=-1).unsqueeze(0)
                final_mask = F.grid_sample(src_mask_t, n_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

            if show_plots:
                plt.figure(figsize=(6, 3))
                plt.plot(losses_log, linewidth=1.5)
                plt.title(f"{model_name} - Level {lvl + 1} Loss Curve")
                plt.grid(True, alpha=0.3)
                plt.show()
                show_mask_comparison_poly(final_mask, ref_bbox_sm, poly_mask_t, title=f"{model_name} - End of Level {lvl+1}")

            if lvl < levels - 1:
                with torch.no_grad():
                    base_err = torch.abs(final_mask - ref_bbox_sm) * poly_mask_t
                    if edge_weight_map is not None: base_err = base_err * edge_weight_map
                
                old_len = len(P_ref_np)
                P_ref_np, P_src_np, _ = add_dynamic_points(base_err[0, 0].cpu().numpy(), P_ref_np, P_src_np, final_grid.cpu().numpy(), n_points=dyn_points_per_level, min_dist=dyn_min_dist, error_threshold=dyn_error_threshold)
                print(f"     [Dynamic] Added {len(P_ref_np) - old_len} points in high-error regions.")

        if show_plots:
            ref_crop_np = ref_img[by0:by1, bx0:bx1]
            ref_crop_rs = cv2.resize(ref_crop_np, (work_w_bbox, work_h_bbox))
            show_points_mesh(P_ref_np, ref_crop_rs, warp_mode, current_simplices, f"Optimized Mesh - {model_name}")
            show_displacement_quiver(P_src_original_init, P_src_np[:len(P_src_original_init)], f"Displacements - {model_name}")
            
        return P_ref_np, P_src_np, final_mask[0, 0].cpu().numpy()

    # RUN ENSEMBLE
    P_ref_A, P_src_A, w_mask_A_np = run_optimization(edge_penalty, "Model A (High Edge Penalty)")
    P_ref_B, P_src_B, w_mask_B_np = run_optimization(0.0, "Model B (No Edge Penalty)")


    # ── Phase 5: Smart Performance-Guided NMS & Blending ──────
    print("\n▸ Phase 5: Generating Performance-Guided Hybrid Mesh")
    
    ref_mask_np = ref_bbox_sm[0, 0].cpu().numpy()
    
    # 1. Compute local errors and smooth them heavily
    err_A = np.abs(w_mask_A_np - ref_mask_np) * poly_mask_work_np
    err_B = np.abs(w_mask_B_np - ref_mask_np) * poly_mask_work_np
    
    ksize = work_w_bbox // 4
    if ksize % 2 == 0: ksize += 1
    err_A_sm = cv2.GaussianBlur(err_A, (ksize, ksize), 0)
    err_B_sm = cv2.GaussianBlur(err_B, (ksize, ksize), 0)
    
    # 2. Performance Weight: 1 where A is better, 0 where B is better
    W_perf = err_B_sm / (err_A_sm + err_B_sm + 1e-8)
    
    # 3. Spatial Prior Weight: 1 at edge, 0 at center
    y_grid, x_grid = np.mgrid[0:work_h_bbox, 0:work_w_bbox]
    nx = np.abs(x_grid - work_w_bbox/2) / (work_w_bbox/2)
    ny = np.abs(y_grid - work_h_bbox/2) / (work_h_bbox/2)
    W_spatial = np.maximum(nx, ny)
    
    # 4. Final Blend & Smoothstep (sharpen transitions)
    W_A = 0.6 * W_perf + 0.4 * W_spatial
    W_A = np.clip(W_A, 0, 1)
    W_A = 3 * W_A**2 - 2 * W_A**3  # Smoothstep
    
    if show_plots:
        plt.figure(figsize=(6, 5))
        plt.imshow(W_A, cmap='magma')
        plt.colorbar(label='0 = B (Center) ---> 1 = A (Edge)')
        plt.title('Hybrid Blend Weight Map (W_A)')
        plt.axis('off')
        plt.show()

    # 5. Smart NMS for Reference Points
    pts_all = np.vstack([P_ref_A, P_ref_B])
    px = np.clip(pts_all[:, 0], 0, work_w_bbox - 1).astype(int)
    py = np.clip(pts_all[:, 1], 0, work_h_bbox - 1).astype(int)
    
    scores_A = W_A[py[:len(P_ref_A)], px[:len(P_ref_A)]]
    scores_B = 1.0 - W_A[py[len(P_ref_A):], px[len(P_ref_A):]]
    scores_all = np.concatenate([scores_A, scores_B])
    
    sort_idx = np.argsort(scores_all)[::-1]
    pts_sorted = pts_all[sort_idx]
    
    kept_pts = []
    min_dist_sq = 25.0 # Minimum 5 pixels apart
    
    for pt in pts_sorted:
        if not kept_pts:
            kept_pts.append(pt)
            continue
        dist_sq = np.sum((np.array(kept_pts) - pt)**2, axis=1)
        if np.min(dist_sq) > min_dist_sq:
            kept_pts.append(pt)
            
    P_ref_hybrid = np.array(kept_pts, dtype=np.float32)
    print(f"  Hybrid mesh created with {len(P_ref_hybrid)} control points (via NMS).")
    
    # 6. Evaluate and Blend Mappings
    P_src_A_eval = eval_warp_at_points_np(P_ref_hybrid, P_ref_A, P_src_A, warp_mode)
    P_src_B_eval = eval_warp_at_points_np(P_ref_hybrid, P_ref_B, P_src_B, warp_mode)
    
    hx = np.clip(P_ref_hybrid[:, 0], 0, work_w_bbox - 1).astype(int)
    hy = np.clip(P_ref_hybrid[:, 1], 0, work_h_bbox - 1).astype(int)
    w_hybrid = W_A[hy, hx][:, None] # shape (N, 1)
    
    P_src_hybrid = w_hybrid * P_src_A_eval + (1.0 - w_hybrid) * P_src_B_eval

    if show_plots:
        P_init_hybrid = apply_affine_to_points(P_ref_hybrid / [scale_x, scale_y], extent_result.src_pts, extent_result.ref_pts, bbox)
        ref_crop_np = ref_img[by0:by1, bx0:bx1]
        ref_crop_rs = cv2.resize(ref_crop_np, (work_w_bbox, work_h_bbox))
        hybrid_simplices = scipy.spatial.Delaunay(P_ref_hybrid).simplices if warp_mode == "affine" else None
        
        show_points_mesh(P_ref_hybrid, ref_crop_rs, warp_mode, hybrid_simplices, f"Final Hybrid Mesh ({warp_mode.upper()})")
        show_displacement_quiver(P_init_hybrid, P_src_hybrid, "Hybrid Control Point Displacements")


    # ── Phase 6: Full-Resolution Application ──────────────────────
    print("\n▸ Phase 6: Rendering Full-Resolution Result from Hybrid Mesh")
    
    P_ref_full = P_ref_hybrid.copy().astype(np.float32)
    P_ref_full[:, 0] /= scale_x
    P_ref_full[:, 1] /= scale_y
    
    P_src_hybrid_t = torch.from_numpy(P_src_hybrid).float().to(device)
    
    if warp_mode == "affine":
        idx_f, wt_f, _ = precompute_affine_barycentric(P_ref_full, bh, bw)
        grid_full = evaluate_affine(P_src_hybrid_t, idx_f.to(device), wt_f.to(device))
    else:
        L_inv_f, U_f, P_grid_f = precompute_tps_matrices(torch.from_numpy(P_ref_full).float().to(device), bh, bw, device)
        grid_full = evaluate_tps(P_src_hybrid_t, L_inv_f, U_f, P_grid_f, bh, bw)

    norm_gx = (grid_full[..., 0] / (w_s - 1)) * 2.0 - 1.0
    norm_gy = (grid_full[..., 1] / (h_s - 1)) * 2.0 - 1.0
    norm_grid_full = torch.stack([norm_gx, norm_gy], dim=-1).unsqueeze(0)
    
    poly_full_np = rasterise_polygon_mask(polygon, bh, bw, offset_xy=(bx0, by0))
    poly_full_t = (torch.from_numpy(poly_full_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        warped_rgb = F.grid_sample(src_img_t, norm_grid_full, mode="bilinear", padding_mode="zeros", align_corners=True) * poly_full_t
        warped_mask = F.grid_sample(src_mask_t, norm_grid_full, mode="bilinear", padding_mode="zeros", align_corners=True) * poly_full_t

    canvas_np = np.full_like(ref_img, get_sea_color(ref_img))
    warped_bbox_bgr = cv2.cvtColor((warped_rgb[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    poly_hw = poly_full_np > 127
    region = canvas_np[by0:by1, bx0:bx1]
    region[poly_hw] = warped_bbox_bgr[poly_hw]
    canvas_np[by0:by1, bx0:bx1] = region
    
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
    
    # ── Phase 7: Result Visualization ─────────────────────────────
    if show_plots:
        show_mask_comparison_poly(
            w_mask_full_t[:, :, by0:by1, bx0:bx1], 
            r_mask_full_t[:, :, by0:by1, bx0:bx1], 
            poly_full_t, 
            title=f"Final Ensemble Full-Res Overlap (IoU={final_iou.item():.4f})"
        )
        
        plt.figure(figsize=(12, 6))
        overlay = cv2.addWeighted(ref_img, 0.45, canvas_np, 0.55, 0)
        poly_mask_vis = rasterise_polygon_mask(polygon, h_r, w_r)
        overlay[poly_mask_vis == 0] = ref_img[poly_mask_vis == 0]
        cv2.polylines(overlay, [polygon.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Hybrid Overlay Result (IoU: {final_iou.item():.4f} | Mode: {warp_mode.upper()})")
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