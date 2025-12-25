from dataclasses import dataclass

import cv2


# @dataclass
class KeypointTrackingConfig:
    repr_error: float = 1.5  # Bidirectional consistency threshold (pixels)

    lk_params: dict = {
        "winSize": (23, 23),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
    }


@dataclass
class InitializationConfig:
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 5
    fb_max_dist: float = 1.0
    ransac_threshold: float = 0.5
    ransac_prob: float = 0.999
    min_buffer_size: int = 2
    min_inliers: int = 100
    min_parallax_angle: float = 2.0  # Minimum median angle in degrees
    min_grid_occupancy: int = 15
    # find_initial_features() still used hardcoded parameters


@dataclass
class PoseEstimationConfig:
    ransac_prob: float = 0.999
    repr_error: float = 2.0


# Previous constructor default values:
# max_features: int = 1500
# min_dist: int = 20
@dataclass
class ReplenishmentConfig:
    max_features = 4000
    min_dist = 7


@dataclass
class TriangulationConfig:
    ransac_prob: float = 0.999
    min_pixel_dist: float = 0.0
    filter_threshold: float = 0.06
