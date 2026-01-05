from __future__ import annotations

import json
import logging
import os

from dataclasses import dataclass
from glob import glob

import cv2
import numpy as np

from pydantic import BaseModel


logger = logging.getLogger(__name__)


# =============================================================================
# Algorithm Configs
# =============================================================================


class KeypointTrackingConfig(BaseModel):
    """Configuration for KLT Optical Flow tracking."""

    win_size: tuple[int, int]
    max_level: int
    criteria_max_iter: int
    criteria_epsilon: float
    bidirectional_error: float
    ransac_threshold: float
    ransac_iters: int

    @property
    def criteria(self) -> tuple:
        return (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.criteria_max_iter,
            self.criteria_epsilon,
        )


class InitializationConfig(BaseModel):
    """Configuration for the Bootstrapping/Initialization phase."""

    cheat_mode: bool
    debug_gt_poses_path: str

    lk_win_size: tuple[int, int]
    lk_max_level: int
    lk_criteria_max_iter: int
    lk_criteria_epsilon: float
    fb_max_dist: float

    min_inliers: int
    ransac_threshold: float
    ransac_prob: float

    min_parallax_angle: float
    parallax_factor: float

    min_grid_occupancy: int
    grid_rows: int
    grid_cols: int

    tile_rows: int
    tile_cols: int
    min_init_features: int

    n_features: int
    contrast_threshold: float

    @property
    def lk_criteria(self) -> tuple:
        return (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.lk_criteria_max_iter,
            self.lk_criteria_epsilon,
        )


class PoseEstimationConfig(BaseModel):
    """Configuration for PnP and Motion-Only Bundle Adjustment."""

    ransac_prob: float
    repr_error: float
    iterations_count: int
    pnp_flags: int

    refinement_loss: str
    refinement_f_scale: float


class ReplenishmentConfig(BaseModel):
    """Parameters for detecting new features to maintain tracking stability."""

    max_features: int
    min_dist: int
    quality_level: float
    block_size: int
    mask_radius: int
    use_harris: bool
    harris_k: float
    harris_threshold: float

    grid_rows: int
    grid_cols: int

    cell_cap_multiplier: float
    global_feature_multiplier: float
    min_feature_factor: float


class TriangulationConfig(BaseModel):
    """Configuration for mapping 2D points to 3D."""

    min_pixel_dist: float
    min_angle_deg: float
    max_depth: float
    min_depth: float
    reset_scale: bool


class LocalBundleAdjustmentConfig(BaseModel):
    """Configuration for Local Bundle Adjustment."""

    enable_ba: bool
    window_size: int
    max_nfev: int
    ftol: float
    loss_function: str
    f_scale: float
    max_reproj_error: float


# =============================================================================
# Dataset Config (Runtime, not serialized)
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuration for an image sequence dataset."""

    name: str
    data_path: str
    image_dir: str
    image_pattern: str
    first_frame: int
    last_frame: int
    ground_truth_path: str | None = None
    debug_output: str | None = None


# =============================================================================
# Root Config
# =============================================================================


class Config(BaseModel):
    """Root configuration containing all sub-configurations."""

    keypoint_tracking: KeypointTrackingConfig
    initialization: InitializationConfig
    pose_estimation: PoseEstimationConfig
    replenishment: ReplenishmentConfig
    triangulation: TriangulationConfig
    local_bundle_adjustment: LocalBundleAdjustmentConfig

    # Runtime fields (not serialized)
    _dataset: DatasetConfig | None = None
    _K: np.ndarray | None = None
    _D: np.ndarray | None = None
    _ground_truth: np.ndarray | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def dataset(self) -> DatasetConfig:
        if self._dataset is None:
            raise ValueError(
                "Dataset not set. Use load_config() to load the full config."
            )
        return self._dataset

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise ValueError(
                "K matrix not loaded. Use load_config() to load the full config."
            )
        return self._K

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            return np.zeros(5)
        return self._D

    @property
    def ground_truth(self) -> np.ndarray | None:
        return self._ground_truth

    def get_image_path(self, idx: int) -> str:
        """Get the full path to an image by index."""
        img_dir = os.path.join(self.dataset.data_path, self.dataset.image_dir)
        if self.dataset.name == "malaga":
            images = sorted(glob(os.path.join(img_dir, self.dataset.image_pattern)))
            filename = images[idx]
        else:
            filename = self.dataset.image_pattern.format(idx)
        return os.path.join(img_dir, filename)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


# =============================================================================
# Config Loader
# =============================================================================


def _load_matrix(path: str, default: np.ndarray | None = None) -> np.ndarray | None:
    """Load a matrix from a text file."""

    if not os.path.exists(path):
        logger.debug("Matrix file not found: %s", path)
        return default

    with open(path) as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        line = line.strip().rstrip(",")
        if not line:
            continue
        values = [float(v.strip()) for v in line.split(",") if v.strip()]
        rows.append(values)

    return np.array(rows)


def _load_ground_truth(path: str) -> np.ndarray | None:
    """Load ground truth poses from file."""

    if not os.path.exists(path):
        logger.debug("Ground truth file not found: %s", path)
        return None

    full_gt = np.loadtxt(path)
    gt = full_gt[:, [-9, -1]]
    logger.debug("Loaded ground truth: %d poses from %s", len(gt), path)
    return gt


def load_config(
    config_path: str,
    dataset: DatasetConfig,
) -> Config:
    """
    Load algorithm configuration from JSON and combine with dataset config.

    Args:
        config_path: Path to the algorithm config JSON file
        dataset: DatasetConfig with runtime parameters

    Returns:
        Fully loaded Config object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = Config.model_validate(json.load(f))

    config._dataset = dataset

    config_dir = os.path.dirname(config_path)
    k_path = os.path.join(config_dir, "K.txt")
    d_path = os.path.join(config_dir, "D.txt")

    config._K = _load_matrix(k_path)
    if config._K is None:
        raise FileNotFoundError(f"Camera matrix K.txt not found in {config_dir}")

    config._D = _load_matrix(d_path, default=np.zeros(5))

    if dataset.ground_truth_path:
        gt_path = os.path.join(dataset.data_path, dataset.ground_truth_path)
        config._ground_truth = _load_ground_truth(gt_path)

    logger.info("Loaded config from %s", config_path)
    logger.debug("Camera matrix K:\n%s", config.K)
    logger.debug("Distortion coefficients D: %s", config.D)

    return config
