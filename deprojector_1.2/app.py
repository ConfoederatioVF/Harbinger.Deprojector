import os
import warnings
import cv2

warnings.filterwarnings("ignore")
EXECUTE_PIPELINE = True

from pipeline.A_coastline_masking import (
    normalize_pair, visualize_normalized_pair, visualize_segmentation
)
from pipeline.B_get_extent import (
    draw_extent, find_extent, visualize_correspondences,
    visualize_full, visualize_ransac, visualize_warp_grid
)

# Import the new point-based optimization function
from pipeline.C_mesh_warp import sgd_point_warp_polygon_constrained

def executePipeline():
    ref_path = "to_projection.png"
    src_path = "from_projection.png"
    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ref_path) or not os.path.exists(src_path):
        raise FileNotFoundError("Ensure 'to_projection.png' and 'from_projection.png' exist.")

    ref_img = cv2.imread(ref_path)
    src_img = cv2.imread(src_path)

    print(f"Reference: {ref_img.shape}")
    print(f"Source:    {src_img.shape}\n")

    # Pre-process: segment and normalise
    pair = normalize_pair(
        ref_img, src_img, work_dim=840, n_clusters=5, verbose=True
    )

    visualize_segmentation(ref_img, pair.ref_seg, title="Reference")
    visualize_segmentation(src_img, pair.src_seg, title="Source")
    visualize_normalized_pair(pair)

    # Run extent finder
    result = find_extent(
        ref_img, src_img, precomputed_pair=pair, mask_threshold=35,
        work_max_dim=840, loftr_dim=640, poly_degree=2,
        ransac_iters=5000, ransac_inlier_frac=0.025, use_anchor=True, verbose=True
    )

    visualize_full(ref_img, src_img, result)
    visualize_correspondences(ref_img, src_img, result, n_show=80)
    visualize_ransac(ref_img, result)

    if result.warp_coeffs is not None:
        visualize_warp_grid(src_img.shape, ref_img, result.warp_coeffs)

    output = draw_extent(ref_img, result, thickness=4)
    cv2.imwrite(os.path.join(output_dir, "extent.png"), output)
    print("\nSaved extent.png")

    # Polygon-constrained control point mesh warp optimisation
    warped_result, learned_disp, detected_bbox, final_iou = (
        sgd_point_warp_polygon_constrained(
            show_plots=True,
            ref_img=ref_img,
            src_img=src_img,
            extent_result=result,
            work_h_bbox=420,
            warp_mode="tps",   
            levels=5,             
            steps_per_lvl=1200,    
            lr_init=2.0,          
            lam_fold=0.2,          
            dyn_points_per_level=24,          
            dyn_error_threshold=0.005,         
            dyn_min_dist=12,
            prune_interval=150,     # Attempt to strip down redundant points safely
            edge_penalty=0,       #16.0: strong edge penalty; 1.0 no edge penalty. 8.0 best compromise?
        )
    )

    cv2.imwrite(os.path.join(output_dir, "output.png"), warped_result)
    print(f"\nSaved output.png (polygon IoU = {final_iou:.4f})")
    cv2.destroyAllWindows()

if __name__ == "__main__" and EXECUTE_PIPELINE == True:
    executePipeline()