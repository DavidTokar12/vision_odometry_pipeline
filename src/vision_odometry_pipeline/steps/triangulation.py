from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_configs import TriangulationConfig
from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class TriangulationStep(VoStep):
    def __init__(self, K: np.ndarray):
        super().__init__("Triangulation")
        self.config = TriangulationConfig()
        self.K = K
        self.max_cos_angle = np.cos(np.radians(self.config.min_angle_deg))

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
        Triangulates candidates and UPDATES the full arrays (Stacking/Removal).
        """
        if len(state.C) == 0:
            return (
                state.P,
                state.X,
                state.landmark_ids,
                state.C,
                state.F,
                state.T_first,
                None,
            )

        # Pre-calculation
        T_first_all = state.T_first.reshape(-1, 3, 4)

        # P for current image
        T_WC_curr = state.pose
        R_CW_curr = T_WC_curr[:3, :3].T
        t_CW_curr = -R_CW_curr @ T_WC_curr[:3, 3]
        M2 = self.K @ np.hstack((R_CW_curr, t_CW_curr.reshape(3, 1)))

        # Mask of candidates to KEEP (default: True).
        # We set to False if:
        #   1. Successfully triangulated (moved to P)
        #   2. Bad point (behind camera / infinity / low parallax / error)
        keep_mask = np.ones(len(state.C), dtype=bool)

        # Candidates worth attempting (sufficient pixel motion)
        displacements = np.linalg.norm(state.C - state.F, axis=1)
        ready_indices = np.where(displacements > self.config.min_pixel_dist)[0]

        new_X_list, new_P_list = [], []

        for idx in ready_indices:
            # P for first image
            T_WC_first = T_first_all[idx]
            R_CW_first = T_WC_first[:3, :3].T
            t_CW_first = -R_CW_first @ T_WC_first[:3, 3]
            M1 = self.K @ np.hstack((R_CW_first, t_CW_first.reshape(3, 1)))

            # Triangulate
            pt1 = state.F[idx].reshape(2, 1)
            pt2 = state.C[idx].reshape(2, 1)
            point_4d = cv2.triangulatePoints(M1, M2, pt1, pt2)

            # Filter points at infinity
            if abs(point_4d[3]) < 1e-6:
                keep_mask[idx] = False  # Garbage
                continue

            # Angle Check (Parallax)
            X = (point_4d[:3] / point_4d[3].flatten()).flatten()
            v1 = X - T_WC_first[:3, 3]
            v2 = X - T_WC_curr[:3, 3]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            cos_angle = np.dot(v1, v2) / (n1 * n2)

            # Clip to prevent numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg < self.config.min_angle_deg:
                continue

            # Cheirality (Behind camera?) and Max Depth Check
            X_local_curr = R_CW_curr @ X + t_CW_curr.flatten()
            if (
                X_local_curr[2] < self.config.min_depth
                or X_local_curr[2] > self.config.max_depth
            ):
                keep_mask[idx] = False
                continue

            X_local_first = R_CW_first @ X + t_CW_first.flatten()
            if (
                X_local_first[2] < self.config.min_depth
                or X_local_first[2] > self.config.max_depth
            ):
                keep_mask[idx] = False
                continue

            # Success!
            new_X_list.append(X)
            new_P_list.append(state.C[idx])
            keep_mask[idx] = False  # Remove from C, moved to P

        # Update P and X
        if len(new_X_list) > 0:
            full_P = np.vstack([state.P, np.array(new_P_list)])
            full_X = np.vstack([state.X, np.array(new_X_list)])

            # Find the max ID currently in use to ensure uniqueness
            start_id = 0
            if len(state.landmark_ids) > 0:
                start_id = state.landmark_ids.max() + 1

            num_new = len(new_X_list)
            new_ids = np.arange(start_id, start_id + num_new, dtype=np.int64)
            full_ids = np.concatenate([state.landmark_ids, new_ids])
        else:
            full_P = state.P
            full_X = state.X
            full_ids = state.landmark_ids

        # Update C, F, T (Keep remaining)
        rem_C = state.C[keep_mask]
        rem_F = state.F[keep_mask]
        rem_T = state.T_first[keep_mask]

        vis = None
        
        if debug:
            # Visualize: Yellow = Waiting, Green = Triangulated this frame
            vis = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_GRAY2BGR)
            for pt in rem_C:  # Waiting
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
            for pt in new_P_list:  # Triangulated
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), 2)

        return full_P, full_X, full_ids, rem_C, rem_F, rem_T, vis

    def _visualize_candidates(self, img, candidates, ready_mask):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pt in candidates[~ready_mask]:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in candidates[ready_mask]:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), 2)
        return vis
