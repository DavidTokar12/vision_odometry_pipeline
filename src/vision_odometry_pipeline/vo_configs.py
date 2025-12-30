from dataclasses import dataclass

import cv2


@dataclass
class KeypointTrackingConfig:
    """Configuration for KLT Optical Flow tracking."""

    win_size: tuple[int, int] = (23, 23)  # Window size for LK optical flow
    max_level: int = 5  # Number of pyramid levels
    # Termination criteria: (Type, Max_Iter, Epsilon)
    criteria: tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)

    bidirectional_error: float = (
        2.0  # Threshold for forward-backward consistency check (pixels)
    )


@dataclass
class InitializationConfig:
    """Configuration for the Bootstrapping/Initialization phase."""

    # --- Change Bootstrap Option ---
    # Set the correct path matching the option in main.py
    CHEATMODE = False
    DEBUG_GT_POSES_PATH = "data/parking/poses.txt"
    # DEBUG_GT_POSES_PATH = "data/kitti/poses/05.txt"

    # --- Tracking during Init ---
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 5
    lk_criteria: tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    fb_max_dist: float = 1.0  # Stricter consistency check for initialization

    # --- Geometric Validation ---
    min_inliers: int = 100  # Minimum valid 3D points to accept initialization
    ransac_threshold: float = 0.5  # Pixel error threshold for Essential Matrix RANSAC
    ransac_prob: float = 0.999  # RANSAC confidence

    # --- Parallax / Baseline ---
    min_parallax_angle: float = 2.0  # Global median parallax angle required (degrees)
    parallax_factor: float = (
        0.5  # Multiplier for individual point check (0.5 * min_parallax)
    )

    # --- Spatial Distribution ---
    min_grid_occupancy: int = 15  # Minimum number of occupied grid cells
    grid_rows: int = 10  # Rows for spatial bucket check
    grid_cols: int = 10  # Cols for spatial bucket check

    # --- Feature Distriubtion (SIFT Tiling) ---
    # Used in find_initial_features to force spread
    tile_rows: int = 4
    tile_cols: int = 4
    min_init_features: int = 15  # Hard fail if fewer features found in frame 0

    # --- Feature Detection (SIFT) ---
    n_features: int = 100  # Target number of features per tile during init
    contrast_threshold: float = (
        0.03  # SIFT contrast threshold (lower = more features, less stable)
    )


@dataclass
class PoseEstimationConfig:
    """Configuration for PnP and Motion-Only Bundle Adjustment."""

    # --- P3P RANSAC ---
    ransac_prob: float = 0.999
    repr_error: float = 2.0  # Max reprojection error for PnP inliers (pixels)
    iterations_count: int = 500  # Max RANSAC iterations
    pnp_flags: int = cv2.SOLVEPNP_P3P  # PnP Method (P3P is fast for minimal sets)

    # --- Least Square ----
    # Robust loss function (linear for errors > f_scale), default is "linear"
    # Options: 'linear' (default/brittle), 'huber', 'soft_l1', 'cauchy', 'arctan'
    refinement_loss: str = "huber"
    # Outlier threshold in pixels. Ignored if loss is 'linear'.
    refinement_f_scale: float = 1.0


@dataclass
class ReplenishmentConfig:
    """
    Parameters for detecting new features to maintain tracking stability.
    """

    # --- Feature Detection (Shi-Tomasi/Harris) ---
    max_features: int = 1000  # Target total number of active features in the system
    min_dist: int = 7  # Minimum pixel distance between features
    quality_level: float = 0.01  # Corner quality level (0.0 to 1.0)
    block_size: int = 3  # Block size for corner computation
    mask_radius: int = (
        6  # Radius around existing points to mask out (usually same as min_dist)
    )
    use_harris: bool = (
        False  # Use Harris detector instead of Shi-Tomasi (more selective)
    )
    harris_k: float = 0.04  # Harris detector free parameter
    harris_threshold: float = 0.1  # Minimum Harris response (filters weak corners)

    # --- Spatial Distribution (Bucketing) ---
    grid_rows: int = 7
    grid_cols: int = 7

    cell_cap_multiplier = 1.5
    global_feature_multiplier = 6


@dataclass
class TriangulationConfig:
    """Configuration for mapping 2D points to 3D."""

    # --- Candidate Selection ---
    min_pixel_dist: float = (
        0.0  # Min pixel displacement before attempting triangulation
    )

    # --- Geometric Filtering ---
    min_angle_deg: float = 1.7  # Minimum triangulation angle (degrees)
    filter_threshold: float = (
        0.05  # Translation threshold to skip angle check (forward motion assumption)
    )
    max_depth: float = (
        200.0  # Maximum allowed depth (meters) to prevent unstable points
    )
    min_depth: float = 1e-6  # Points must be in front of camera
