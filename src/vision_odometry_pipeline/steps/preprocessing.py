from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class ImagePreprocessingStep(VoStep):
    def __init__(self) -> None:
        super().__init__("ImagePreprocessing")

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Converts the current image to grayscale and removes distortion.

        Returns:
            Normal: (gray_undistorted_image,)
            Debug:  (gray_undistorted_image, visualization_image)
        """
        EqualizeHist = False  # Sometimes helps KLT tracking

        # Input is the raw current image
        # Note: We assume the Runner ensures 'curr' is not None before calling steps
        img = state.image_buffer.curr

        # Logic: Convert to grayscale
        gray: np.ndarray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Logic: Remove distortion
        map_x, map_y, roi = state.
        #this maps need to be computed only once, at the beginning of the pipeline.
        # see the additional function in folder "steps" called "create_undistorted_maps.py"
        # that creates this maps.
        # Run the function in main at the beginning and store the maps and roi in something that
        #  can be quickly accessed by steps.

        gray_undistorted = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        x, y, w, h = roi
        gray_undistorted = gray_undistorted[y:y+h, x:x+w]

        # Optional: Equalize Hist (often helps KLT)
        if EqualizeHist:
            gray_undistorted = cv2.equalizeHist(gray_undistorted)

        if debug:
            # Visualization: Just the grayscale image
            return gray_undistorted, gray_undistorted

        return gray_undistorted, None
