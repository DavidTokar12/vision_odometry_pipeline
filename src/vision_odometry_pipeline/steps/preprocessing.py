from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class ImagePreprocessingStep(VoStep):
    def __init__(self, map_x: np.ndarray, map_y: np.ndarray, roi) -> None:
        super().__init__("ImagePreprocessing")
        self.map_x = map_x
        self.map_y = map_y
        self.roi = roi

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        img = state.image_buffer.curr

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray_undistorted = cv2.remap(
            gray, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR
        )

        # Crop ROI
        x, y, w, h = self.roi
        gray_undistorted = gray_undistorted[y : y + h, x : x + w]

        # Optional: Equalize Hist
        if True:
            gray_undistorted = cv2.equalizeHist(gray_undistorted)

        if debug:
            return gray_undistorted, gray_undistorted

        return gray_undistorted, None
