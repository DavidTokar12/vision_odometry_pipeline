from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class PoseEstimationConfig:
    ransac_prob: float = 0.999
    repr_error: float = 2.0


class PoseEstimationStep(VoStep):
    def __init__(self, K: np.ndarray):
        super().__init__("PoseEstimation")
        self.config = PoseEstimationConfig()
        self.K = K

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Estimates Pose and FILTERS outliers from P and X.

        Returns:
            (New_Pose, Inlier_P, Inlier_X, Vis)
        """

        if len(state.P) < 4:
            if debug:
                vis_fail = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_GRAY2BGR)
                return state.pose, state.P, state.X, vis_fail
            return state.pose, state.P, state.X, None

        # P3P RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            state.X,  # Triangulated 3D Landmarks
            state.P,  # Tracked 2D Keypoints
            self.K,
            None,  # No distorsion
            iterationsCount=100,
            reprojectionError=self.config.repr_error,
            confidence=self.config.ransac_prob,
            flags=cv2.SOLVEPNP_P3P,  # use P3P, need 4 points
        )

        new_pose = state.pose.copy()

        # Filter Outliers
        # -------------------------------------
        if success:
            # Convert vector to 3x3 matrix
            R, _ = cv2.Rodrigues(rvec)
            # world to camera transform
            T_CW = np.eye(4)
            T_CW[:3, :3] = R
            T_CW[:3, 3] = tvec.flatten()
            # camera to world transform
            T_WC = np.linalg.inv(T_CW)
            new_pose = T_WC

            if inliers is not None:
                mask = inliers.flatten()
                final_P = state.P[mask]  # Keep only inlier 2D points
                final_X = state.X[mask]  # Keep only inlier 3D points
            else:
                final_P = state.P
                final_X = state.X
        else:
            final_P = state.P
            final_X = state.X

        # Visualization
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
