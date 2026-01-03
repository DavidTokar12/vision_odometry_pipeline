from __future__ import annotations

import logging
import os

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from glob import glob

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for an image sequence dataset."""

    name: str
    K: np.ndarray
    image_path_fn: Callable[[int], str]
    last_frame: int
    first_frame: int = 0
    D: np.ndarray = field(default_factory=lambda: np.zeros(5))
    debug_output: str | None = None
    ground_truth: np.ndarray | None = None

    def __post_init__(self):
        self.K = np.asarray(self.K)
        self.D = np.asarray(self.D)

        if self.first_frame >= self.last_frame:
            raise ValueError(
                f"first_frame ({self.first_frame}) must be < last_frame ({self.last_frame})"
            )


class ImageSequence:
    """
    Iterator over a sequence of images.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.current_idx = config.first_frame

        if config.debug_output:
            os.makedirs(config.debug_output, exist_ok=True)
            logger.debug("Created output directory: %s", config.debug_output)

        logger.info(
            "ImageSequence initialized: %s (frames %d-%d)",
            config.name,
            config.first_frame,
            config.last_frame,
        )
        logger.debug("Camera matrix K:\n%s", config.K)
        logger.debug("Distortion coefficients D: %s", config.D)

    @property
    def K(self) -> np.ndarray:
        return self.config.K

    @property
    def D(self) -> np.ndarray:
        return self.config.D

    @property
    def ground_truth(self) -> np.ndarray | None:
        return self.config.ground_truth

    @property
    def debug_output(self) -> str | None:
        return self.config.debug_output

    @property
    def first_frame(self) -> int:
        return self.config.first_frame

    @property
    def last_frame(self) -> int:
        return self.config.last_frame

    @property
    def is_finished(self) -> bool:
        return self.current_idx >= self.config.last_frame

    def peek_image(self) -> np.ndarray | None:
        """Get current image WITHOUT advancing the counter."""
        if self.is_finished:
            return None

        img_path = self.config.image_path_fn(self.current_idx)
        print(img_path)
        return cv2.imread(img_path)

    def get_image(self) -> np.ndarray | None:
        """Get the next image in the sequence."""
        if self.is_finished:
            return None

        img_path = self.config.image_path_fn(self.current_idx)
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
        self.current_idx = self.config.first_frame
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
        return self.config.last_frame - self.config.first_frame


# =============================================================================
# Factory Functions for Known Datasets
# =============================================================================


def _load_ground_truth(
    path: str, columns: tuple[int, int] = (-9, -1)
) -> np.ndarray | None:
    """Load ground truth from a poses file."""
    if not os.path.exists(path):
        logger.debug("Ground truth file not found: %s", path)
        return None

    full_gt = np.loadtxt(path)
    gt = full_gt[:, list(columns)]
    logger.debug("Loaded ground truth: %d poses from %s", len(gt), path)
    return gt


def create_parking_config(
    base_path: str,
    output: str,
    first_frame: int = 0,
    last_frame: int | None = None,
) -> DatasetConfig:
    """Create configuration for the Parking dataset."""
    data_path = os.path.join(base_path, "parking")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Parking dataset not found: {data_path}")

    K = np.eye(3)
    k_path = os.path.join(data_path, "K.txt")
    if os.path.exists(k_path):
        K = np.loadtxt(k_path, delimiter=",", usecols=(0, 1, 2))

    gt_path = os.path.join(data_path, "poses.txt")
    ground_truth = _load_ground_truth(gt_path)

    return DatasetConfig(
        name="Parking",
        K=K,
        image_path_fn=lambda idx: os.path.join(
            data_path, "images", f"img_{idx:05d}.png"
        ),
        first_frame=first_frame,
        last_frame=last_frame if last_frame is not None else 598,
        debug_output=os.path.join(output, "main_parking"),
        ground_truth=ground_truth,
    )


def create_kitti_config(
    base_path: str,
    output: str,
    first_frame: int = 0,
    last_frame: int | None = None,
    sequence: str = "05",
) -> DatasetConfig:
    """Create configuration for the KITTI dataset."""
    data_path = os.path.join(base_path, "kitti")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"KITTI dataset not found: {data_path}")

    K = np.array(
        [
            [7.18856e02, 0, 6.071928e02],
            [0, 7.18856e02, 1.852157e02],
            [0, 0, 1],
        ]
    )

    gt_path = os.path.join(data_path, "poses", f"{sequence}.txt")
    ground_truth = _load_ground_truth(gt_path)

    return DatasetConfig(
        name=f"KITTI-{sequence}",
        K=K,
        image_path_fn=lambda idx: os.path.join(
            data_path, sequence, "image_0", f"{idx:06d}.png"
        ),
        first_frame=first_frame,
        last_frame=last_frame if last_frame is not None else 2760,
        debug_output=os.path.join(output, "main_kitti"),
        ground_truth=ground_truth,
    )


def create_malaga_config(
    base_path: str,
    output: str,
    first_frame: int = 0,
    last_frame: int | None = None,
) -> DatasetConfig:
    """Create configuration for the Malaga dataset."""
    data_path = os.path.join(base_path, "malaga-urban-dataset-extract-07")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Malaga dataset not found: {data_path}")

    img_dir = os.path.join(
        data_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images"
    )
    image_paths = sorted(glob(os.path.join(img_dir, "*left.jpg")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    K = np.array(
        [
            [621.18428, 0, 404.0076],
            [0, 621.18428, 309.05989],
            [0, 0, 1],
        ]
    )

    return DatasetConfig(
        name="Malaga",
        K=K,
        image_path_fn=lambda idx: image_paths[idx] if idx < len(image_paths) else "",
        first_frame=first_frame,
        last_frame=last_frame if last_frame is not None else len(image_paths),
        debug_output=os.path.join(output, "main_malaga"),
        ground_truth=None,
    )


def create_polibahn_up_config(
    base_path: str,
    output: str,
    first_frame: int = 0,
    last_frame: int | None = None,
) -> DatasetConfig:
    """Create configuration for the Polibahn Up dataset."""
    data_path = os.path.join(base_path, "polibahn_up")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Polibahn dataset not found: {data_path}")

    k_path = os.path.join(data_path, "K.txt")
    d_path = os.path.join(data_path, "D.txt")

    K = np.loadtxt(k_path, delimiter=",") if os.path.exists(k_path) else np.eye(3)
    D = np.loadtxt(d_path, delimiter=",") if os.path.exists(d_path) else np.zeros(5)

    gt_path = os.path.join(data_path, "poses.txt")
    ground_truth = _load_ground_truth(gt_path)

    return DatasetConfig(
        name="Polibahn_Up",
        K=K,
        D=D,
        image_path_fn=lambda idx: os.path.join(
            data_path, "images", f"img_{idx:05d}.png"
        ),
        first_frame=first_frame,
        last_frame=last_frame if last_frame is not None else 462,
        debug_output=os.path.join(output, "main_polibahn_up"),
        ground_truth=ground_truth,
    )


DATASET_FACTORIES = {
    0: create_parking_config,
    1: create_kitti_config,
    2: create_malaga_config,
    3: create_polibahn_up_config,
    "parking": create_parking_config,
    "kitti": create_kitti_config,
    "malaga": create_malaga_config,
    "polibahn_up": create_polibahn_up_config,
}


def create_config(
    dataset: int | str,
    base_path: str,
    output: str,
    first_frame: int = 0,
    last_frame: int | None = None,
    **kwargs,
) -> DatasetConfig:
    if dataset not in DATASET_FACTORIES:
        raise ValueError(
            f"Unknown dataset: {dataset}. Valid: {list(DATASET_FACTORIES.keys())}"
        )

    factory = DATASET_FACTORIES[dataset]
    return factory(
        base_path=base_path,
        output=output,
        first_frame=first_frame,
        last_frame=last_frame,
        **kwargs,
    )
