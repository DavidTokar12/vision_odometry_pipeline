from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_step import VoStep


class ImagePreprocessingStep(VoStep):
    def __init__(self, map_x: np.ndarray, map_y: np.ndarray, roi) -> None:
        super().__init__("ImagePreprocessing")
        self.map_x = map_x
        self.map_y = map_y
        self.roi = roi

    def process(
        self, img: np.ndarray, debug: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray_undistorted = cv2.remap(
            gray, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR
        )

        # Crop ROI
        x, y, w, h = self.roi
        gray_undistorted = gray_undistorted[y : y + h, x : x + w]

        # Different Preprocessing Methods
        preproc_clahe = False
        preproc_gamma = False
        preproc_log = False
        preproc_hist = False
        preproc_bilat = False

        if preproc_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_undistorted = clahe.apply(gray_undistorted)

            # Optional: Mild blur to reduce noise amplified by CLAHE
            gray_undistorted = cv2.GaussianBlur(gray_undistorted, (3, 3), 0)

        if preproc_gamma:
            gamma = 1.5  # Values < 1.0 brighten the image
            look_up_table = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")
            gray_undistorted = cv2.LUT(gray_undistorted, look_up_table)

        if preproc_log:
            # Convert to float for precision
            img_float = gray_undistorted.astype(np.float32)

            # Apply Log Transform: s = c * log(1 + r)
            # We calculate 'c' such that the maximum pixel value maps to 255
            c = 255 / np.log(1 + np.max(img_float))
            log_image = c * (np.log(img_float + 1))

            # Convert back to uint8
            gray_undistorted = np.array(log_image, dtype=np.uint8)

        if preproc_hist:
            gray_undistorted = cv2.equalizeHist(gray_undistorted)

        if preproc_bilat:
            gray_undistorted = cv2.bilateralFilter(gray_undistorted, 15, 15, 15)

        if debug:
            return gray_undistorted, gray_undistorted

        return gray_undistorted, None
