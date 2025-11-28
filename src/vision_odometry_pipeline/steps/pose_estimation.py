from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class PoseEstimationStep(VoStep):
    def __init__(self, K: np.ndarray):
        super().__init__("PoseEstimation")
        self.K = K

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Estimates Pose and FILTERS outliers from P and X.

        Returns:
            (New_Pose, Inlier_P, Inlier_X, Vis)
        """
        # 1. Validation
        if len(state.P) < 4:
            if debug:
                vis_fail = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_GRAY2BGR)
                return state.pose, state.P, state.X, vis_fail
            return state.pose, state.P, state.X, None

        # 2. P3P RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            state.X,
            state.P,
            self.K,
            None,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_P3P,
        )

        new_pose = state.pose.copy()

        # 3. Filter Outliers (Logic Moved Here)
        # -------------------------------------
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T_CW = np.eye(4)
            T_CW[:3, :3] = R
            T_CW[:3, 3] = tvec.flatten()
            new_pose = np.linalg.inv(T_CW)

            if inliers is not None:
                mask = inliers.flatten()
                final_P = state.P[mask]
                final_X = state.X[mask]
            else:
                final_P = state.P
                final_X = state.X
        else:
            final_P = state.P
            final_X = state.X

        # 4. Visualization
        vis = None
        if debug:
            if success and inliers is not None:
                vis = self._visualize_reprojection(
                    state.image_buffer.curr, final_P, final_X, rvec, tvec
                )
            else:
                vis = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_GRAY2BGR)

        return new_pose, final_P, final_X, vis

    def _visualize_reprojection(self, img, p_in, x_in, rvec, tvec):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(x_in) == 0:
            return vis
        projected, _ = cv2.projectPoints(x_in, rvec, tvec, self.K, None)
        for p_meas, p_proj in zip(p_in, projected, strict=False):
            cv2.circle(vis, (int(p_meas[0]), int(p_meas[1])), 2, (0, 255, 0), -1)
            cv2.circle(vis, (int(p_proj[0][0]), int(p_proj[0][1])), 2, (0, 0, 255), -1)
        return vis
