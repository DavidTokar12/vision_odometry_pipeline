from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from vision_odometry_pipeline.vo_state import VoState


class VoStep(ABC):
    """
    Abstract Base Class for all VO Pipeline Steps.
    Enforces inputs are treated as read-only.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, state: VoState, debug: bool) -> tuple[Any]:
        """
        Core Logic.

        Args:
            state: The input state (Read-Only).
                   Do NOT modify arrays in-place.
            debug: If true creates a debug visualization for the given step.
        Returns:
            Any number of any objects that are the results of this step,
            if debug is true the image object is returned as the last parameter
        """
