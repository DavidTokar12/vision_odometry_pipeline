from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class KeypointTrackingStep(VoStep):
    def __init__(self, lk_params: dict | None = None):
        super().__init__("KeypointTracking")

        self.lk_params = lk_params or {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
    ]:
        """
        Tracks points using KLT and FILTERS the state arrays based on tracking status.

        Returns:
            (New_P, New_X, New_C, New_F, New_T_first, Vis)
        """
        img_prev = state.image_buffer.prev
        img_curr = state.image_buffer.curr

        if img_prev is None or img_curr is None:
            raise ValueError("Tracking requires two images in buffer")

        # 1. Track Active Keypoints (P)
        p0 = state.P.astype(np.float32)
        p1 = np.empty((0, 2), dtype=np.float32)
        st_p = np.empty((0,), dtype=np.uint8)

        if len(p0) > 0:
            # p1, st_p, _ = cv2.calcOpticalFlowPyrLK(
            #     img_prev, img_curr, p0, None, **self.lk_params
            # )
            p1, st_p = self._track_features_bidirectional(img_prev, img_curr, p0)
            st_p = st_p.reshape(-1)

        # 2. Track Candidate Keypoints (C)
        c0 = state.C.astype(np.float32)
        c1 = np.empty((0, 2), dtype=np.float32)
        st_c = np.empty((0,), dtype=np.uint8)

        if len(c0) > 0:
            # c1, st_c, _ = cv2.calcOpticalFlowPyrLK(
            #     img_prev, img_curr, c0, None, **self.lk_params
            # )
            c1, st_c = self._track_features_bidirectional(img_prev, img_curr, c0)
            st_c = st_c.reshape(-1)

        # 3. Filter Data
        # --------------------------------
        # Filter P and align X
        valid_p = st_p == 1
        new_P = p1[valid_p]
        new_X = state.X[valid_p]

        # Filter C and align F, T
        valid_c = st_c == 1
        new_C = c1[valid_c]
        new_F = state.F[valid_c]
        new_T = state.T_first[valid_c]

        # 4. Visualization
        vis = None
        if debug:
            vis = self._visualize_tracking(img_curr, p0, p1, st_p, c0, c1, st_c)

        return new_P, new_X, new_C, new_F, new_T, vis

    def _visualize_tracking(
        self,
        img: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        st_p: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
        st_c: np.ndarray,
    ) -> np.ndarray:
        """
        Private helper to visualize tracking lines.
        Green = Active Keypoints (P)
        Blue  = Candidate Keypoints (C)
        """
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw P (Green)
        if len(p0) > 0:
            for new, old, good in zip(p1, p0, st_p, strict=True):
                if good:
                    cv2.line(
                        vis,
                        (int(new[0]), int(new[1])),
                        (int(old[0]), int(old[1])),
                        (0, 255, 0),
                        2,
                    )

        # Draw C (Blue)
        if len(c0) > 0:
            for new, old, good in zip(c1, c0, st_c, strict=True):
                if good:
                    cv2.line(
                        vis,
                        (int(new[0]), int(new[1])),
                        (int(old[0]), int(old[1])),
                        (255, 0, 0),
                        2,
                    )

        return vis

    def _track_features_bidirectional(self, img0, img1, p0):
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # Backward flow
        p0r, st2, err2 = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        # Check consistency (L-infinity norm)
        dist = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good_mask = (
            (st1.flatten() == 1)
            & (st2.flatten() == 1)
            & (dist < 1.0)  # this was in the config file
        )

        return p1, good_mask
