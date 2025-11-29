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
        Converts the current image to grayscale.

        Returns:
            Normal: (gray_image,)
            Debug:  (gray_image, visualization_image)
        """

        # Input is the raw current image
        # Note: We assume the Runner ensures 'curr' is not None before calling steps
        img = state.image_buffer.curr

        # Logic: Convert to grayscale
        gray: np.ndarray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Optional: Equalize Hist (often helps KLT)
        gray = cv2.equalizeHist(gray)

        if debug:
            # Visualization: Just the grayscale image
            return gray, gray

        return gray, None
