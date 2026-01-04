from __future__ import annotations

import logging

import cv2
import numpy as np

from vision_odometry_pipeline.vo_configs import Config


logger = logging.getLogger(__name__)


class ImageSequence:
    """
    Iterator over a sequence of images.
    """

    def __init__(self, config: Config):
        self.config = config
        self.current_idx = config.dataset.first_frame

        logger.info(
            "ImageSequence initialized: %s (frames %d-%d)",
            config.dataset.name,
            config.dataset.first_frame,
            config.dataset.last_frame,
        )

    @property
    def is_finished(self) -> bool:
        return self.current_idx >= self.config.dataset.last_frame

    def peek_image(self) -> np.ndarray | None:
        """Get current image WITHOUT advancing the counter."""
        if self.is_finished:
            return None

        img_path = self.config.get_image_path(self.current_idx)
        return cv2.imread(img_path)

    def get_image(self) -> np.ndarray | None:
        """Get the next image in the sequence."""
        if self.is_finished:
            return None

        img_path = self.config.get_image_path(self.current_idx)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(
                "Failed to load frame %d from %s", self.current_idx, img_path
            )
        else:
            logger.debug("Loaded frame %d: %s", self.current_idx, img_path)

        self.current_idx += 1
        return img

    def reset(self) -> None:
        """Reset to first frame."""
        self.current_idx = self.config.dataset.first_frame
        logger.debug("Reset to frame %d", self.current_idx)

    def __iter__(self) -> ImageSequence:
        self.reset()
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self.is_finished:
            raise StopIteration

        frame_id = self.current_idx
        image = self.get_image()

        if image is None:
            raise StopIteration

        return frame_id, image

    def __len__(self) -> int:
        return self.config.dataset.last_frame - self.config.dataset.first_frame
