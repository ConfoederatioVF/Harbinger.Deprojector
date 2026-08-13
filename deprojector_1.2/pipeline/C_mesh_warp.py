"""
Pipeline Step C: Point-Constrained Mesh Warp (v3 — Affine Triangles & TPS)
Optimizes a sparse set of control points using PyTorch SGD. Supports both 
Affine Triangulation (barycentric rendering) and Thin Plate Spline (TPS) modes.
Points are dynamically added to high-error regions during segmentation.
"""

from typing import Tuple, Optional, Any, Callable, List

import cv2
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
    # Convert working-bbox coords to full coords for matrix calc
    full_pts = pts.copy()
    full_pts[:, 0] += bbox["x0"]
    full_pts[:, 1] += bbox["y0"]
    
    M, _ = cv2.estimateAffine2D(ref_pts.astype(np.float32), src_pts.astype(np.float32), method=cv2.RANSAC)
    if M is None:
        return pts.copy() # Fallback to identity mapping if affine fails
        
    ones = np.ones((len(full_pts), 1))
    pts_h = np.hstack([full_pts, ones])
    mapped = (M @ pts_h.T).T
    return mapped.astype(np.float32)

def add_dynamic_points(
    error_map_np: np.ndarray, P_ref_np: np.ndarray, P_src_np: np.ndarray, 
    current_grid_np: np.ndarray, n_points: int = 5, min_dist: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds high-error regions and inserts new control points there."""
    err = cv2.GaussianBlur(error_map_np.astype(np.float32), (9, 9), 0)
    new_refs, new_srcs = [], []
    
    for _ in range(n_points):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(err)
        if max_val < 0.05: # Error threshold
            break
            
        # Ensure it's not too close to existing points
        dist = np.linalg.norm(P_ref_np - np.array(max_loc), axis=1)
        if np.any(dist < min_dist):
            cv2.circle(err, max_loc, min_dist, 0, -1)
            continue
            
        new_refs.append([max_loc[0], max_loc[1]])
        new_srcs.append(current_grid_np[max_loc[1], max_loc[0]]) # Map to its current source interpolation
        cv2.circle(err, max_loc, min_dist, 0, -1)
        
    if new_refs:
        P_ref_np = np.vstack([P_ref_np, new_refs])
        P_src_np = np.vstack([P_src_np, new_srcs])
        
    return P_ref_np, P_src_np

# ─────────────────────────────────────────────────────────────────────
# 3. DIFFERENTIABLE WARPING KERNELS (AFFINE & TPS)
# ─────────────────────────────────────────────────────────────────────

def precompute_affine_barycentric(P_ref_np: np.ndarray, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Precomputes Delaunay Triangulation and Barycentric weights for every pixel."""
    delaunay = scipy.spatial.Delaunay(P_ref_np)
    y, x = np.mgrid[0:H, 0:W]
    grid_pts = np.column_stack([x.ravel(), y.ravel()])
    
    simplices = delaunay.find_simplex(grid_pts)
    
    # Scipy transform matrix for barycentric coords: T_inv @ (P - C)
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
    """Differentiable Affine Triangulation forward pass."""
    # (H, W, 3, 2)
    src_triangles = P_src[indices] 
    # (H, W, 2) = Sum of (vertices * weights)
    return (src_triangles * weights.unsqueeze(-1)).sum(dim=2)

def precompute_tps_matrices(P_ref: torch.Tensor, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precomputes Thin Plate Spline invariant matrices."""
    N = P_ref.shape[0]
    dist = torch.cdist(P_ref, P_ref)
    K = dist ** 2 * torch.log(dist + 1e-8)
    P = torch.cat([torch.ones(N, 1, device=device), P_ref], dim=1)
    
    L = torch.zeros(N + 3, N + 3, device=device)
    L[:N, :N] = K
    L[:N, N:] = P
    L[N:, :N] = P.t()
    L = L + torch.eye(N + 3, device=device) * 1e-5 # Reg to prevent singularities
    
    L_inv = torch.linalg.pinv(L)
    
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_pts = torch.stack([x.flatten(), y.flatten()], dim=1).float()
    
    dist_grid = torch.cdist(grid_pts, P_ref)
    U = dist_grid ** 2 * torch.log(dist_grid + 1e-8)
    P_grid = torch.cat([torch.ones(H*W, 1, device=device), grid_pts], dim=1)
    
    return L_inv, U, P_grid

def evaluate_tps(P_src: torch.Tensor, L_inv: torch.Tensor, U: torch.Tensor, P_grid: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Differentiable TPS forward pass."""
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
    """Penalizes inverted triangles (where signed area becomes negative)."""
    A = P_src[simplices[:, 0]]
    B = P_src[simplices[:, 1]]
    C = P_src[simplices[:, 2]]
    # Cross product
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    return F.relu(-area + 0.1).mean()

def bend_loss_tps(P_src: torch.Tensor, L_inv: torch.Tensor, N: int) -> torch.Tensor:
    """Penalizes extreme TPS warp bending."""
    Y = torch.cat([P_src, torch.zeros(3, 2, device=P_src.device)], dim=0)
    coeffs = L_inv @ Y
    W_mat = coeffs[:N, :]
    return (W_mat ** 2).mean()

def dice_loss_poly(pred: torch.Tensor, target: torch.Tensor, poly_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = (pred * poly_mask).reshape(-1)
    t = (target * poly_mask).reshape(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)

def mse_loss_poly(pred: torch.Tensor, target: torch.Tensor, poly_mask: torch.Tensor) -> torch.Tensor:
    n = poly_mask.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * poly_mask).sum() / n

# ─────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def show_points_mesh(P_ref_np: np.ndarray, ref_img_np: np.ndarray, simplices: Optional[np.ndarray] = None, title: str = "Mesh"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(ref_img_np, cv2.COLOR_BGR2RGB))
    
    if simplices is not None:
        ax.triplot(P_ref_np[:, 0], P_ref_np[:, 1], simplices, color='cyan', lw=1.5, alpha=0.8)
    
    ax.plot(P_ref_np[:, 0], P_ref_np[:, 1], 'yo', markersize=4)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────
# 6. MAIN SGD POINT OPTIMIZATION PIPELINE
# ─────────────────────────────────────────────────────────────────────

def sgd_point_warp_polygon_constrained(
    ref_img: np.ndarray,
    src_img: np.ndarray,
    extent_result: ExtentResult,
    work_h_bbox: int = 384,
    warp_mode: str = "affine", # 'affine' or 'tps'
    levels: int = 4,
    steps_per_lvl: int = 350,
    lr_init: float = 2.0,
    lam_fold: float = 0.5,
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
    
    work_w_bbox = int(work_h_bbox * (bw / bh))
    scale_x, scale_y = work_w_bbox / bw, work_h_bbox / bh
    
    # ── Phase 2: Masks & Tensors ──────────────────────────────────
    mask_ref_full = extract_land_mask(ref_img)
    mask_src_full = extract_land_mask(src_img)
    
    poly_mask_work_np = rasterise_polygon_mask(polygon, work_h_bbox, work_w_bbox, offset_xy=(bx0, by0), scale_xy=(scale_x, scale_y))
    poly_mask_t = (torch.from_numpy(poly_mask_work_np).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    ref_bbox_t = (torch.from_numpy(mask_ref_full[by0:by1, bx0:bx1]).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    ref_bbox_sm = F.interpolate(ref_bbox_t, (work_h_bbox, work_w_bbox), mode="bilinear", align_corners=True)
    
    src_mask_t = (torch.from_numpy(mask_src_full).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    src_rgb_np = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img_t = (torch.from_numpy(src_rgb_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)
    
    # ── Phase 3: Control Point Initialization ─────────────────────
    print("\n▸ Phase 3: Initializing Control Points")
    P_ref_np = init_control_points(bbox, poly_mask_work_np, grid_size=6)
    P_src_np = apply_affine_to_points(P_ref_np / [scale_x, scale_y], extent_result.src_pts, extent_result.ref_pts, bbox)
    
    print(f"  Initialized {len(P_ref_np)} control points.")
    
    # ── Phase 4: Dynamic SGD Optimization Loop ────────────────────
    print(f"\n▸ Phase 4: Multi-Level {warp_mode.upper()} SGD Point Optimization")
    
    current_simplices = None
    
    for lvl in range(levels):
        lr = lr_init * (0.6 ** lvl)
        print(f"  ── Level {lvl + 1}/{levels} | Points: {len(P_ref_np)} | LR: {lr:.4f}")
        
        # Precompute mappings for current topology
        if warp_mode == "affine":
            idx_t, wt_t, current_simplices = precompute_affine_barycentric(P_ref_np, work_h_bbox, work_w_bbox)
            idx_t, wt_t = idx_t.to(device), wt_t.to(device)
        else:
            L_inv, U, P_grid = precompute_tps_matrices(torch.from_numpy(P_ref_np).to(device), work_h_bbox, work_w_bbox, device)
            
        P_src_t = torch.tensor(P_src_np, dtype=torch.float32, device=device, requires_grad=True)
        optim = torch.optim.Adam([P_src_t], lr=lr)
        
        best_loss = float('inf')
        best_P_src = P_src_np.copy()
        
        # SGD Loop
        for step in range(steps_per_lvl):
            optim.zero_grad()
            
            # Forward Pass: Build Grid
            if warp_mode == "affine":
                grid_src = evaluate_affine(P_src_t, idx_t, wt_t)
            else:
                grid_src = evaluate_tps(P_src_t, L_inv, U, P_grid, work_h_bbox, work_w_bbox)
                
            # Normalize for grid_sample: [-1, 1]
            norm_grid_x = (grid_src[..., 0] / (w_s - 1)) * 2.0 - 1.0
            norm_grid_y = (grid_src[..., 1] / (h_s - 1)) * 2.0 - 1.0
            norm_grid = torch.stack([norm_grid_x, norm_grid_y], dim=-1).unsqueeze(0)
            
            # Warp Src Mask
            warped = F.grid_sample(src_mask_t, norm_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            
            l_dice = dice_loss_poly(warped, ref_bbox_sm, poly_mask_t)
            l_mse = mse_loss_poly(warped, ref_bbox_sm, poly_mask_t)
            
            # Structural/Fold Loss
            if warp_mode == "affine":
                l_struct = fold_loss_affine(P_src_t, current_simplices)
            else:
                l_struct = bend_loss_tps(P_src_t, L_inv, len(P_ref_np))
                
            loss = l_dice + l_mse + lam_fold * l_struct
            loss.backward()
            optim.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_P_src = P_src_t.detach().cpu().numpy()
                
            if (step + 1) % 100 == 0 or step == steps_per_lvl - 1:
                print(f"     step {step+1:4d} | loss={loss.item():.4f} | dice={l_dice.item():.4f} | struct={l_struct.item():.4f}")

        # Post-level updates
        P_src_np = best_P_src.copy()
        
        # Add Dynamic Points if not the last level
        if lvl < levels - 1:
            with torch.no_grad():
                if warp_mode == "affine":
                    final_grid = evaluate_affine(torch.from_numpy(P_src_np).to(device), idx_t, wt_t)
                else:
                    final_grid = evaluate_tps(torch.from_numpy(P_src_np).to(device), L_inv, U, P_grid, work_h_bbox, work_w_bbox)
                    
                n_gx = (final_grid[..., 0] / (w_s - 1)) * 2.0 - 1.0
                n_gy = (final_grid[..., 1] / (h_s - 1)) * 2.0 - 1.0
                n_grid = torch.stack([n_gx, n_gy], dim=-1).unsqueeze(0)
                
                w_mask = F.grid_sample(src_mask_t, n_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                error_map = (torch.abs(w_mask - ref_bbox_sm) * poly_mask_t)[0, 0].cpu().numpy()
                
            old_len = len(P_ref_np)
            P_ref_np, P_src_np = add_dynamic_points(error_map, P_ref_np, P_src_np, final_grid.cpu().numpy(), n_points=8, min_dist=15)
            print(f"     [Dynamic] Added {len(P_ref_np) - old_len} points in high-error regions.")

    # Show final local mesh
    ref_crop_np = ref_img[by0:by1, bx0:bx1]
    ref_crop_rs = cv2.resize(ref_crop_np, (work_w_bbox, work_h_bbox))
    show_points_mesh(P_ref_np, ref_crop_rs, current_simplices, f"Optimized Mesh ({warp_mode.upper()})")

    # ── Phase 5: Full-Resolution Application ──────────────────────
    print("\n▸ Phase 5: Rendering Full-Resolution Result")
    
    P_ref_full = P_ref_np.copy()
    P_ref_full[:, 0] /= scale_x
    P_ref_full[:, 1] /= scale_y
    
    if warp_mode == "affine":
        idx_f, wt_f, _ = precompute_affine_barycentric(P_ref_full, bh, bw)
        grid_full = evaluate_affine(torch.from_numpy(P_src_np).to(device), idx_f.to(device), wt_f.to(device))
    else:
        L_inv_f, U_f, P_grid_f = precompute_tps_matrices(torch.from_numpy(P_ref_full).to(device), bh, bw, device)
        grid_full = evaluate_tps(torch.from_numpy(P_src_np).to(device), L_inv_f, U_f, P_grid_f, bh, bw)

    norm_gx = (grid_full[..., 0] / (w_s - 1)) * 2.0 - 1.0
    norm_gy = (grid_full[..., 1] / (h_s - 1)) * 2.0 - 1.0
    norm_grid_full = torch.stack([norm_gx, norm_gy], dim=-1).unsqueeze(0)
    
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

    print("\n✓ Done.")
    return canvas_np, grid_full.cpu().numpy(), bbox, final_iou.item()