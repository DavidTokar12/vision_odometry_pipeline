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


class InitialPoseFindingStep(VoStep):
    def __init__(self) -> None:
        super().__init__("InitialPoseFindingStep")

    def process(self, state: VoState, debug: bool):  # TODO: Add return values
        # TODO: Implement

        return

    def find_initial_features(self, state: VoState):
        # 3. Feature Detection (SIFT on the FIRST frame of the buffer)
        img0 = state.image_buffer.curr
        sift = cv2.SIFT_create()
        kp0 = sift.detect(img0, None)

        if not kp0:
            print("Init: No keypoints found.")
            return img0, None

        # Format points: (N, 1, 2)
        p_initial = np.float32([kp.pt for kp in kp0]).reshape(-1, 1, 2)
        return p_initial
