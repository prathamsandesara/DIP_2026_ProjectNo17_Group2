"""
Submission: Complete Multi-Branch Weave-Adaptive Fabric Defect Detection Pipeline.
Strictly pure Python code implementing the multi-branch Digital Image Processing pipeline.
"""

from .utils import (
    get_logger,
    load_image_gray,
    load_mask_binary,
    overlay_mask_on_image,
    resolve_n_workers,
)
from .step00_dataset_setup import load_one_class_split, build_one_class_splits
from .step01_preprocessing import (
    preprocess_pipeline,
    alpha_trimmed_mean_filter,
    clahe_enhance,
    detect_active_fabric_mask,
    get_support_valid_mask,
)
from .step02_pitch_and_scale_estimation import (
    get_adaptive_footprints,
    get_dual_scale_kernels,
    estimate_weave_pitch,
)
from .step03_micro_branch import run_micro_branch, compute_dual_scale_micro
from .step04_macro_branch import run_macro_branch, compute_macro_residuals
from .step05_warp_branch import run_warp_branch
from .step06_texture_branch import run_texture_branch, compute_lbp, build_gabor_bank
from .step07_calibration_and_normal_model import calibrate_all_fabrics, calibrate_fabric_thresholds
from .step08_multimodal_feature_fusion import fuse_base_branches, fuse_all_features
from .step09_morphological_cleanup import filter_small_components, apply_macro_geometry_closing, cleanup_mask
from .step10_localization_and_bounding_boxes import extract_defect_regions, draw_defect_boxes_on_image
from .step11_operational_decision_and_screening import (
    evaluate_single_roll,
    aggregate_operational_metrics,
    print_evaluation_summary_table,
)
from .step12_end_to_end_pipeline import run_pipeline, inspect_single_fabric_roll

__all__ = [
    "get_logger",
    "load_image_gray",
    "load_mask_binary",
    "overlay_mask_on_image",
    "resolve_n_workers",
    "load_one_class_split",
    "build_one_class_splits",
    "preprocess_pipeline",
    "alpha_trimmed_mean_filter",
    "clahe_enhance",
    "detect_active_fabric_mask",
    "get_support_valid_mask",
    "get_adaptive_footprints",
    "get_dual_scale_kernels",
    "estimate_weave_pitch",
    "run_micro_branch",
    "compute_dual_scale_micro",
    "run_macro_branch",
    "compute_macro_residuals",
    "run_warp_branch",
    "run_texture_branch",
    "compute_lbp",
    "build_gabor_bank",
    "calibrate_all_fabrics",
    "calibrate_fabric_thresholds",
    "fuse_base_branches",
    "fuse_all_features",
    "filter_small_components",
    "apply_macro_geometry_closing",
    "cleanup_mask",
    "extract_defect_regions",
    "draw_defect_boxes_on_image",
    "evaluate_single_roll",
    "aggregate_operational_metrics",
    "print_evaluation_summary_table",
    "run_pipeline",
    "inspect_single_fabric_roll",
]
