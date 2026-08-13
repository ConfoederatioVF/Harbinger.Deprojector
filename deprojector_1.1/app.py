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
from pipeline.C_mesh_warp import sgd_mesh_warp_polygon_constrained

def executePipeline ():
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

  # Polygon-constrained mesh warp optimisation
  warped_result, learned_disp, detected_bbox, final_iou = (
    sgd_mesh_warp_polygon_constrained(
      ref_img=ref_img,
      src_img=src_img,
      extent_result=result,
      work_h_bbox=384,
      cell_schedule=(48, 24, 12, 6, 4, 2),
      steps_per_lvl=(600, 600, 600, 600, 800, 800),
      lr_schedule=(5e-3, 3e-3, 1.5e-3, 8e-4, 3e-4, 1e-4),
      lam_bend_schedule=(0.5, 1.0, 2.0, 6.0, 15.0, 40.0),
      lam_residual=(0.0, 0.01, 0.03, 0.08, 0.2, 0.4),
      lam_fold=0.2,
      lam_jac=0.1,
      lam_leakage=5.0,
      proj_sigma_schedule=(2.0, 1.5, 1.0, 0.6, 0.35, 0.2),
      proj_every=50,
      max_disp_schedule=(0.6, 0.65, 0.7, 0.75, 0.8, 0.85),
      hard_clamp_every=10,
    )
  )

  cv2.imwrite(os.path.join(output_dir, "output.png"), warped_result)
  print(f"\nSaved output.png (polygon IoU = {final_iou:.4f})")
  cv2.destroyAllWindows()

if __name__ == "__main__" and EXECUTE_PIPELINE == True:
  executePipeline()