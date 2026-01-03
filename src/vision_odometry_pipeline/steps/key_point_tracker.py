from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_configs import KeypointTrackingConfig
from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class KeypointTrackingStep(VoStep):
    def __init__(self):
        super().__init__("KeypointTracking")

        self.config = KeypointTrackingConfig()

        self.lk_params = {
            "winSize": self.config.win_size,
            "maxLevel": self.config.max_level,
            "criteria": self.config.criteria,
        }

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
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

        # Track Active Keypoints (P)
        p0 = state.P
        p1 = np.empty((0, 2), dtype=np.float32)
        st_p = np.empty((0,), dtype=bool)

        if len(p0) > 0:
            p1, st_p = self._track_features_bidirectional(img_prev, img_curr, p0)
            st_p = st_p.reshape(-1)

        # Track Candidate Keypoints (C)
        c0 = state.C
        c1 = np.empty((0, 2), dtype=np.float32)
        st_c = np.empty((0,), dtype=bool)

        if len(c0) > 0:
            c1, st_c = self._track_features_bidirectional(img_prev, img_curr, c0)
            st_c = st_c.reshape(-1)

        """ (TODO: potentially remove)
        # Filter Data (without 8-Point RANSAC) 
        # --------------------------------
        # Filter P and align X
        valid_p = st_p == 1
        new_P = p1[valid_p]
        new_X = state.X[valid_p]
        new_ids = state.landmark_ids[valid_p]

        # Filter C and align F, T
        valid_c = st_c == 1
        new_C = c1[valid_c]
        new_F = state.F[valid_c]
        new_T = state.T_first[valid_c]
        """

        # --- Filter Data (With 8-Point RANSAC) ---
        # (TODO: potentially remove or clean up)

        # Gather all successfully tracked points for 8-Point RANSAC
        # Convert boolean masks to indices to easily map RANSAC results back
        idx_p_good = np.flatnonzero(st_p)
        idx_c_good = np.flatnonzero(st_c)

        # Prepare data for RANSAC (Prev -> Curr)
        pts_prev_all = np.concatenate((p0[idx_p_good], c0[idx_c_good]), axis=0).astype(
            np.float32, copy=False
        )
        pts_curr_all = np.concatenate((p1[idx_p_good], c1[idx_c_good]), axis=0).astype(
            np.float32, copy=False
        )

        # Run 8-Point RANSAC
        if len(pts_prev_all) >= 8:
            _, ransac_mask = cv2.findFundamentalMat(
                pts_prev_all,
                pts_curr_all,
                cv2.FM_RANSAC,
                self.config.ransac_threshold,
                0.99,  # confidence
                self.config.ransac_iters,
            )
            ransac_mask = ransac_mask.flatten() == 1
        else:
            ransac_mask = np.ones(len(pts_prev_all), dtype=bool)

        # Split mask back to P and C components
        split_idx = len(idx_p_good)
        mask_p_ransac = ransac_mask[:split_idx]
        mask_c_ransac = ransac_mask[split_idx:]

        # Apply combined filter (KLT Status AND RANSAC Inlier)
        final_idx_p = idx_p_good[mask_p_ransac]
        final_idx_c = idx_c_good[mask_c_ransac]

        # Filter P and align X
        new_P = p1[final_idx_p]
        new_X = state.X[final_idx_p]
        new_ids = state.landmark_ids[final_idx_p]

        # Filter C and align F, T
        new_C = c1[final_idx_c]
        new_F = state.F[final_idx_c]
        new_T = state.T_first[final_idx_c]

        # 4. Visualization
        vis = None
        if debug:
            vis = self._visualize_tracking(img_curr, p0, p1, st_p, c0, c1, st_c)

        return new_P, new_X, new_ids, new_C, new_F, new_T, vis

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
        # Ensure float32 and OpenCV-friendly shape (N,1,2)
        p0 = np.asarray(p0, dtype=np.float32).reshape(-1, 1, 2)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)

        # Back to (N,2) for your pipeline
        p0_2 = p0.reshape(-1, 2)
        p1_2 = p1.reshape(-1, 2)
        p0r_2 = p0r.reshape(-1, 2)

        dist = np.abs(p0_2 - p0r_2).max(axis=1)  # stays float32
        good_mask = (
            (st1.reshape(-1) == 1)
            & (st2.reshape(-1) == 1)
            & (dist < self.config.bidirectional_error)
        )

        return p1_2.astype(np.float32, copy=False), good_mask
