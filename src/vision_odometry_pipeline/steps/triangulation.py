from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class TriangulationConfig:
    ransac_prob: float = 0.999
    min_pixel_dist: float = 0.0
    filter_threshold: float = 0.06


class TriangulationStep(VoStep):
    def __init__(self, K: np.ndarray, min_angle_deg: float = 1.0):
        super().__init__("Triangulation")
        self.config = TriangulationConfig()
        self.K = K
        self.min_angle_deg = min_angle_deg
        self.max_cos_angle = np.cos(np.radians(min_angle_deg))

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
    ]:
        """
        Triangulates candidates and UPDATES the full arrays (Stacking/Removal).

        Returns:
            (Full_P, Full_X, Remaining_C, Remaining_F, Remaining_T, Vis)
        """
        if len(state.C) == 0:
            if debug:
                vis = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_GRAY2BGR)
                return state.P, state.X, state.C, state.F, state.T_first, vis
            return state.P, state.X, state.C, state.F, state.T_first, None

        # --- Selection Logic ---

        # now this is doing nothing, everything is filtered after. Might still be useful in the future
        displacements = np.linalg.norm(state.C - state.F, axis=1)
        # Create selection mask: TRUE if pixel moved enough
        ready_mask = displacements > self.config.min_pixel_dist

        # Needed for triangulation math below
        T_first_all = state.T_first.reshape(-1, 3, 4)

        # --- Triangulation Loop ---
        new_X_list, new_P_list = [], []
        indices = np.where(ready_mask)[0]

        if len(indices) > 0:
            # P for current image
            T_WC_curr = state.pose
            R_CW_curr = T_WC_curr[:3, :3].T
            t_CW_curr = -R_CW_curr @ T_WC_curr[:3, 3]
            M2 = self.K @ np.hstack((R_CW_curr, t_CW_curr.reshape(3, 1)))

            # P for first image
            for idx in indices:
                T_WC_first = T_first_all[idx]
                R_CW_first = T_WC_first[:3, :3].T
                t_CW_first = -R_CW_first @ T_WC_first[:3, 3]

                M1 = self.K @ np.hstack((R_CW_first, t_CW_first.reshape(3, 1)))

                # first observation pixel coords and candidate 2D keypoint, triangulate
                pt1 = state.F[idx].reshape(2, 1)
                pt2 = state.C[idx].reshape(2, 1)
                point_4d = cv2.triangulatePoints(M1, M2, pt1, pt2)

                # Filter points at infinity
                if abs(point_4d[3]) < 1e-6:
                    continue

                # homogeneous coordinate of the candidate 3D landmark
                X = (point_4d[:3] / point_4d[3].flatten()).flatten()

                # Check if motion is completely in front of you. If yes, don't apply angle filter
                if (
                    abs(T_WC_curr[:3, 3] - T_WC_first[:3, 3])[0]
                    > self.config.filter_threshold
                ):
                    # Triangulation angle check
                    ray1 = X - T_WC_first[:3, 3]
                    ray2 = X - T_WC_curr[:3, 3]

                    cos_angle = np.dot(ray1, ray2) / (
                        np.linalg.norm(ray1) * np.linalg.norm(ray2)
                    )
                    # Reject if angle too small
                    if abs(cos_angle) >= self.max_cos_angle:
                        continue

                # Cheirality check (Is point in front of camera?)
                X_local = R_CW_curr @ X + t_CW_curr.flatten()
                # Discard point if it is behind the camera
                if X_local[2] < 0:
                    continue

                # Stricter depth filtering to prevent scale drift
                if X_local[2] > 300:
                    continue

                new_X_list.append(X)
                new_P_list.append(state.C[idx])

        # --- Data Mutation (Logic Moved Here) ---
        # 1. Update P and X
        if len(new_X_list) > 0:
            new_X_arr = np.array(new_X_list)
            new_P_arr = np.array(new_P_list)
            full_P = np.vstack([state.P, new_P_arr])
            full_X = np.vstack([state.X, new_X_arr])
        else:
            full_P = state.P
            full_X = state.X

        # 2. Update Candidates (Keep those NOT ready)
        mask_keep = ~ready_mask
        rem_C = state.C[mask_keep]
        rem_F = state.F[mask_keep]
        rem_T = state.T_first[mask_keep]

        vis = None
        if debug:
            vis = self._visualize_candidates(
                state.image_buffer.curr, state.C, ready_mask
            )

        return full_P, full_X, rem_C, rem_F, rem_T, vis

    def _visualize_candidates(self, img, candidates, ready_mask):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pt in candidates[~ready_mask]:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in candidates[ready_mask]:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), 2)
        return vis
