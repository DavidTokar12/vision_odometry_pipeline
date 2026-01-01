from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import field

import numpy as np


class ImageSlidingWindow:
    def __init__(self):
        self._buffer = deque(maxlen=2)

    def update(self, new_image: np.ndarray):
        """
        Push new image.
        - If buffer was empty: [New]
        - If buffer had 1: [Old, New]
        - If buffer had 2: [Old_Old (dropped), Old (becomes prev), New (becomes curr)]
        Reference copy only. Zero data allocation.
        """
        self._buffer.append(new_image)

    @property
    def curr(self) -> np.ndarray:
        if not self._buffer:
            raise ValueError("Buffer empty")
        return self._buffer[-1]

    @property
    def prev(self) -> np.ndarray:
        if len(self._buffer) < 2:
            return None  # Handle first frame edge case
        return self._buffer[0]

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) == 2


@dataclass
class VoState:
    """
    Pure Data Class.
    Holds ONLY the mathematical state required for the Markovian process.
    """

    # Image Memory (Reference Swapping)
    image_buffer: ImageSlidingWindow = field(default_factory=ImageSlidingWindow)

    # P^i: Tracked 2D Keypoints [N, 2]
    P: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    # X^i: Triangulated 3D Landmarks [N, 3]
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))

    # C^i: Candidate 2D Keypoints [M, 2]
    C: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    # F^i: First observation pixel coords [M, 2]
    F: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    # T_first: Camera Pose at first observation [M, 12] (Flattened 3x4)
    T_first: np.ndarray = field(
        default_factory=lambda: np.empty((0, 12), dtype=np.float32)
    )
    # ID^i: Unique ID for every tracked point [N,]
    landmark_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )

    # T_WC: Current Camera Pose [4, 4]
    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))

    # The average depth of landmarks observed at initialization.
    initial_avg_depth: float = 1.0

    frame_id: int = 0

    pipline_init_stage: int = 0
