from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


# --- Configuration ---
@dataclass
class InitializationConfig:
    lk_win_size: tuple[int, int] = (15, 15)
    lk_max_level: int = 3
    fb_max_dist: float = 1.0
    ransac_threshold: float = 1.0
    ransac_prob: float = 0.999
    min_buffer_size: int = 2
    min_inliers: int = 8


class PipelineInitialization(VoStep):
    def __init__(self, K, D) -> None:
        super().__init__("PipelineInitialization")
        self.initial_K = K
        self.initial_D = D

    def process(self, state: VoState, debug: bool):  # TODO: Add return values
        # TODO: Implement

        return

    def find_initial_features(self, state: VoState):
        # Feature Detection (SIFT on the FIRST frame of the buffer)
        img0 = state.image_buffer.curr
        sift = cv2.SIFT_create()
        kp0 = sift.detect(img0, None)

        if not kp0:
            print("Init: No keypoints found.")
            return img0, None

        # Format points: (N, 1, 2)
        p_initial = np.float32([kp.pt for kp in kp0]).reshape(-1, 1, 2)
        return p_initial

    def create_undistorted_maps(self, image_size):
        """
        Generate lookup maps to remove image distortion.

        Args:
            K: Camera intrinsic matrix (3x3)
            D: Distortion coefficients
            image_size: Tuple of (height, width) for the image resolution

        Returns:
            map_x, map_y: Lookup maps for cv2.remap() to undistort images
            roi: Region of interest after undistortion (x, y, w, h)
        """
        h, w = image_size

        # Compute optimal camera matrix to handle black borders
        # alpha=0: crop all black pixels; alpha=1: keep all original pixels
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.initial_K, self.initial_D, (w, h), alpha=0, newImgSize=(w, h)
        )

        # Generate lookup tables for fast image undistortion
        # CV_16SC2 format is faster and more memory-efficient than CV_32FC1
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.initial_K,
            self.initial_D,
            None,  # R (Rotation matrix) - None for monocular cameras
            new_K,  # New camera matrix with optimal parameters
            (w, h),
            cv2.CV_16SC2,
        )

        return map_x, map_y, roi, new_K
