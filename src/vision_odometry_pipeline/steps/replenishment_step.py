from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class ReplenishmentStep(VoStep):
    def __init__(self, max_features: int = 500, min_dist: int = 10):
        super().__init__("Replenishment")
        self.max_features = max_features
        self.min_dist = min_dist

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Detects new corners and STACKS them onto the state arrays.

        Returns:
            (Full_C, Full_F, Full_T, Vis)
        """
        curr_img = state.image_buffer.curr

        # 1. Masking
        mask = np.full(curr_img.shape, 255, dtype=np.uint8)
        all_pts = np.vstack([state.P, state.C]) if len(state.C) > 0 else state.P
        if len(all_pts) > 0:
            for pt in all_pts:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), self.min_dist, 0, -1)

        # 2. Detection
        n_needed = self.max_features - len(all_pts)

        if not n_needed:
            return state.C, state.F, state.T_first, None

        # Feature Detection (SIFT on the FIRST frame of the buffer)
        sift = cv2.SIFT_create()
        sift_keypoints = sift.detect(curr_img, mask)

        keypoints = np.array(
            [kp.pt for kp in sift_keypoints], dtype=np.float32
        ).reshape(-1, 2)
        if len(keypoints) > n_needed:
            keypoints = keypoints[:n_needed]

        # identity_pose_flat = np.hstack((np.eye(3), np.zeros((3, 1)))).flatten()
        # T_first_init = np.tile(identity_pose_flat, (len(keypoints), 1))

        # if n_needed > 0:
        #     pts = cv2.goodFeaturesToTrack(
        #         curr_img,
        #         mask=mask,
        #         maxCorners=n_needed,
        #         qualityLevel=0.01,
        #         minDistance=self.min_dist,
        #     )
        #     if pts is not None:
        #         new_candidates = pts.reshape(-1, 2)

        # 3. Data Stacking (Logic Moved Here)
        # -----------------------------------
        if len(keypoints) > 0:
            full_C = np.vstack([state.C, keypoints])
            full_F = np.vstack([state.F, keypoints])  # F is current pixel loc

            # Tile the current pose for the new candidates
            # Assuming state.pose is 4x4, we flatten the top 3x4 (R|t) to 12
            pose_3x4 = state.pose[:3, :].flatten()
            tiled_poses = np.tile(pose_3x4, (len(keypoints), 1))
            full_T = np.vstack([state.T_first, tiled_poses])
        else:
            full_C = state.C
            full_F = state.F
            full_T = state.T_first

        # 4. Visualization
        vis = None
        if debug:
            vis = self._visualize_new_features(curr_img, mask, keypoints)

        return full_C, full_F, full_T, vis

    def _visualize_new_features(self, img, mask, new_pts):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[mask == 0] = vis[mask == 0] // 2
        for pt in new_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
        return vis
