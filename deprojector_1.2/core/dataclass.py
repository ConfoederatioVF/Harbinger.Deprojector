import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

@dataclass
class ExtentResult:
    polygon: Optional[np.ndarray]
    n_correspondences: int
    confidence: float
    method: str
    src_pts: Optional[np.ndarray] = None
    ref_pts: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    warp_coeffs: Optional[dict] = None
    debug: dict = field(default_factory=dict)

@dataclass
class SegmentationResult:
    """Output of the land/sea segmentation pipeline."""
    land_mask: np.ndarray          # uint8, 255=land, 0=sea
    sea_mask: np.ndarray           # uint8, 255=sea, 0=land
    coastline: np.ndarray          # uint8, 255=coastline pixels
    sea_color_lab: np.ndarray      # mean sea color in Lab space
    land_color_lab: np.ndarray     # mean land color in Lab space
    confidence: float              # segmentation confidence [0,1]
    debug: dict = field(default_factory=dict)


@dataclass
class NormalizedPair:
    """Mutually compatible representations for a ref/src pair."""
    ref_land_mask: np.ndarray
    src_land_mask: np.ndarray
    ref_coastline: np.ndarray
    src_coastline: np.ndarray
    ref_dt_signed: np.ndarray      # float32, signed distance
    src_dt_signed: np.ndarray
    ref_coastline_dt: np.ndarray   # float32, distance to coast
    src_coastline_dt: np.ndarray
    ref_curvature: np.ndarray      # float32, coastline curvature
    src_curvature: np.ndarray
    ref_orientation: np.ndarray    # float32, coastline orientation
    src_orientation: np.ndarray
    ref_shape_context: np.ndarray  # multi-channel descriptor image
    src_shape_context: np.ndarray
    ref_seg: SegmentationResult
    src_seg: SegmentationResult