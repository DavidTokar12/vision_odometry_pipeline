import os

from glob import glob

import cv2
import numpy as np


class ImageSequence:
    def __init__(
        self, dataset_id: int, first_frame: int = 0, last_frame: int | None = None
    ):
        """
        Initialize the DataLoader.

        Args:
            dataset_id (int): 0: Parking, 1: KITTI, 2: Malaga
            last_frame (int, optional): Overwrite the default last frame limit.
        """
        self.dataset_id = dataset_id
        self.current_idx = first_frame

        if not first_frame < last_frame:
            raise ValueError("first_frame must by strictly smaller than last_frame")

        # Base paths
        self.base_path = "/workspaces/vision_odometry_pipeline"
        self.kitti_path = os.path.join(self.base_path, "data/kitti")
        self.malaga_path = os.path.join(
            self.base_path, "data/malaga-urban-dataset-extract-07"
        )
        self.parking_path = os.path.join(self.base_path, "data/parking")

        # Default initialization
        self.K = np.eye(3)
        self.D = np.zeros(5)
        self.ground_truth = None
        self.debug_output = ""
        self.images = []  # For list-based datasets (Malaga)

        if dataset_id == 0:  # Parking
            self._setup_parking(last_frame)
        elif dataset_id == 1:  # KITTI
            self._setup_kitti(last_frame)
        elif dataset_id == 2:  # Malaga
            self._setup_malaga(last_frame)
        else:
            raise ValueError(f"Invalid dataset id: {dataset_id}")

        # Ensure output directory exists
        if self.debug_output:
            os.makedirs(self.debug_output, exist_ok=True)

        print(f"DataLoader Initialized [ID: {dataset_id}]")
        print(f"  > Output: {self.debug_output}")
        print(f"  > Camera Matrix K:\n{self.K}")
        print(f"  > Distortion Matrix D:\n{self.D}")

    def _setup_parking(self, last_frame):
        if not os.path.exists(self.parking_path):
            raise FileNotFoundError(f"Parking path not found: {self.parking_path}")
        self.debug_output = os.path.join(self.base_path, "debug_output/main_parking/")

        # Load K
        k_path = os.path.join(self.parking_path, "K.txt")
        if os.path.exists(k_path):
            self.K = np.loadtxt(k_path, delimiter=",", usecols=(0, 1, 2))

        # Load Ground Truth
        gt_path = os.path.join(self.parking_path, "poses.txt")
        if os.path.exists(gt_path):
            full_gt = np.loadtxt(gt_path)
            self.ground_truth = full_gt[:, [-9, -1]]

        self.last_frame = last_frame if last_frame is not None else 598

    def _setup_kitti(self, last_frame):
        if not os.path.exists(self.kitti_path):
            raise FileNotFoundError(f"KITTI path not found: {self.kitti_path}")
        self.debug_output = os.path.join(self.base_path, "debug_output/main_kitti/")

        # KITTI Hardcoded K (from template)
        self.K = np.array(
            [[7.18856e02, 0, 6.071928e02], [0, 7.18856e02, 1.852157e02], [0, 0, 1]]
        )

        # Load Ground Truth
        gt_path = os.path.join(self.kitti_path, "poses", "05.txt")
        if os.path.exists(gt_path):
            full_gt = np.loadtxt(gt_path)
            self.ground_truth = full_gt[:, [-9, -1]]  # x and z coordinates

        self.last_frame = last_frame if last_frame is not None else 2760

    def _setup_malaga(self, last_frame):
        if not os.path.exists(self.malaga_path):
            raise FileNotFoundError(f"Malaga path not found: {self.malaga_path}")
        self.debug_output = os.path.join(self.base_path, "debug_output/main_malaga/")

        # Setup Images (Malaga uses weird naming, so we glob)
        img_dir = os.path.join(
            self.malaga_path, "malaga-urban-dataset-extract-07_rectified_800x600_Images"
        )
        self.images = sorted(glob(os.path.join(img_dir, "*left.jpg")))
        if not self.images:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # Malaga K
        self.K = np.array(
            [[621.18428, 0, 404.0076], [0, 621.18428, 309.05989], [0, 0, 1]]
        )

        self.last_frame = last_frame if last_frame is not None else len(self.images)

    @property
    def is_finished(self) -> bool:
        return self.current_idx >= self.last_frame

    def get_image(self) -> np.ndarray:
        """Get the next image in the sequence."""
        if self.is_finished:
            return None

        img = None

        if self.dataset_id == 0:  # Parking
            img_path = os.path.join(
                self.parking_path, "images", f"img_{self.current_idx:05d}.png"
            )
            img = cv2.imread(img_path)

        elif self.dataset_id == 1:  # KITTI
            img_path = os.path.join(
                self.kitti_path, "05", "image_0", f"{self.current_idx:06d}.png"
            )
            img = cv2.imread(img_path)

        elif self.dataset_id == 2:  # Malaga
            if self.current_idx < len(self.images):
                img_path = self.images[self.current_idx]
                img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Failed to load frame {self.current_idx}")

        self.current_idx += 1
        return img
