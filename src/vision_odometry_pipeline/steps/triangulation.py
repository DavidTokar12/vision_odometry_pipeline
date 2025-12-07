from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class TriangulationStep(VoStep):
    def __init__(self, K: np.ndarray, min_angle_deg: float = 2.0):
        super().__init__("Triangulation")
        self.K = K
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.min_angle_cos = np.cos(np.deg2rad(min_angle_deg))

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

        # --- Math Logic (Same as before) ---
        # Unit normalized first image coordinates
        rays_first_cam = np.ones((len(state.C), 3), dtype=np.float32)
        rays_first_cam[:, 0] = (state.F[:, 0] - self.cx) / self.fx
        rays_first_cam[:, 1] = (state.F[:, 1] - self.cy) / self.fy

        # Unit normalized current image coordinates
        rays_curr_cam = np.ones((len(state.C), 3), dtype=np.float32)
        rays_curr_cam[:, 0] = (state.C[:, 0] - self.cx) / self.fx
        rays_curr_cam[:, 1] = (state.C[:, 1] - self.cy) / self.fy

        # Normalize to unit vector
        rays_first_cam /= np.linalg.norm(rays_first_cam, axis=1, keepdims=True)
        rays_curr_cam /= np.linalg.norm(rays_curr_cam, axis=1, keepdims=True)

        # Transform to world frame
        T_first_all = state.T_first.reshape(-1, 3, 4)
        R_first_all = T_first_all[:, :3, :3]
        R_curr = state.pose[:3, :3]

        rays_first_world = np.einsum("nij,nj->ni", R_first_all, rays_first_cam)
        rays_curr_world = rays_curr_cam @ R_curr.T

        # Compute angle between rays and create selection mask
        dot_products = np.sum(rays_first_world * rays_curr_world, axis=1)
        ready_mask = dot_products < self.min_angle_cos

        # --- Triangulation Loop ---
        new_X_list, new_P_list = [], []
        indices = np.where(ready_mask)[0]

        if len(indices) > 0:
            # P for current image
            T_WC_curr = state.pose
            R_CW_curr = T_WC_curr[:3, :3].T
            t_CW_curr = -R_CW_curr @ T_WC_curr[:3, 3]
            P2 = self.K @ np.hstack((R_CW_curr, t_CW_curr.reshape(3, 1)))

            # P for first image
            for idx in indices:
                T_WC_first = T_first_all[idx]
                R_CW_first = T_WC_first[:3, :3].T
                t_CW_first = -R_CW_first @ T_WC_first[:3, 3]
                P1 = self.K @ np.hstack((R_CW_first, t_CW_first.reshape(3, 1)))

                # first observation pixel coords and candidate 2D keypoint, triangulate
                pt1 = state.F[idx].reshape(2, 1)
                pt2 = state.C[idx].reshape(2, 1)
                point_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)

                # Filter points at infinity
                if abs(point_4d[3]) < 1e-6:
                    continue

                # homogeneous coordinate of the candidate 3D landmark
                X = (point_4d[:3] / point_4d[3].flatten()).flatten()

                # Cheirality check (Is point in front of camera?)
                X_local = R_CW_curr @ X + t_CW_curr.flatten()
                # Discard point if it is behind the camera
                if X_local[2] <= 0:
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
