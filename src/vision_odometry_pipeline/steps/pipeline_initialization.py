from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


# --- Configuration ---
@dataclass
class InitializationConfig:
    lk_win_size: tuple[int, int] = (15, 15)
    lk_max_level: int = 3
    fb_max_dist: float = 1.0
    ransac_threshold: float = 1.0
    ransac_prob: float = 0.999
    min_buffer_size: int = 2
    min_inliers: int = 8


class PipelineInitialization(VoStep):
    def __init__(self, K, D) -> None:
        super().__init__("PipelineInitialization")
        self.initial_K = K
        self.initial_D = D
        self.optimal_K = None
        self.config = InitializationConfig()

    def process(self, state: VoState, debug: bool):  # TODO: Add return values
        img_prev = state.image_buffer.prev
        img_curr = state.image_buffer.curr

        # Track Keypoints
        p0 = state.C.astype(np.float32)
        p1, st = self._track_features_bidirectional(img_prev, img_curr, p0)
        p1 = p1.reshape(-1, 2)

        # Update Candidates
        st = st.flatten() == 1
        new_C = p1[st]
        new_F = state.F[st]
        new_T = state.T_first[st]

        # Check if points have moved enough
        displacements = np.linalg.norm(new_C - new_F, axis=1)
        displacements.sort()
        displacements = displacements[-30:]

        avg_parallax = np.mean(displacements) if len(displacements) > 0 else 0.0
        if avg_parallax < 20.0:
            return new_C, new_F, new_T, None, None, None, False

        # Compute Essential Matrix (between First Obs F and Current C)
        E, mask = cv2.findEssentialMat(
            new_C,
            new_F,
            self.optimal_K,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold,
        )

        if E is None:
            return new_C, new_F, new_T, None, None, None, False

        # Recover pose with inliers
        mask = mask.ravel()
        pts0 = new_F[mask]
        ptsn = new_C[mask]

        if len(ptsn) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, False

        _, R, t, mask_pose = cv2.recoverPose(E, ptsn, pts0, self.optimal_K)

        # Filter cheirality (keep points in front of camera) and triangulate
        pose_inliers = mask_pose.ravel() > 0
        ptsn = ptsn[pose_inliers]
        pts0 = pts0[pose_inliers]

        if len(ptsn) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, False

        P0 = self.optimal_K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = self.optimal_K @ np.hstack((R, t))
        points4D = cv2.triangulatePoints(P0, P1, pts0.T, ptsn.T)

        pts_hom = points4D[:3]
        W = points4D[3]

        mask_finite = np.abs(W) > 1e-4

        valid_mask = np.zeros(W.shape, dtype=bool)

        if np.any(mask_finite):
            points3D = pts_hom[:, mask_finite] / W[mask_finite]
            in_front = points3D[2] > 0
            valid_mask[mask_finite] = in_front

        num_valid = np.sum(valid_mask)
        if num_valid > self.config.min_inliers:
            print(f"[Init] Success! {num_valid} landmarks.")

            # Construct new state
            new_X = (pts_hom[:, valid_mask] / W[valid_mask]).T
            new_P = ptsn[valid_mask]

            new_pose = np.eye(4)
            new_pose[:3, :3] = R
            new_pose[:3, 3] = t.flatten()

            return new_C, new_F, new_T, new_X, new_P, new_pose, True

        return new_C, new_F, new_T, None, None, None, False

    def find_initial_features(self, state: VoState):
        # Feature Detection (SIFT on the FIRST frame of the buffer)
        img = state.image_buffer.curr
        sift = cv2.SIFT_create()
        sift_keypoints = sift.detect(img, None)

        keypoints = np.array(
            [kp.pt for kp in sift_keypoints], dtype=np.float32
        ).reshape(-1, 2)

        if len(keypoints) < 15:  # TODO: Adjust this threshold
            print("Warning: Low feature count in initialization frame")

        identity_pose_flat = np.hstack((np.eye(3), np.zeros((3, 1)))).flatten()
        T_first_init = np.tile(identity_pose_flat, (len(keypoints), 1))

        return keypoints, keypoints, T_first_init

    def create_undistorted_maps(self, image_size):
        """
        Generate lookup maps to remove image distortion.

        Args:
            K: Camera intrinsic matrix (3x3)
            D: Distortion coefficients
            image_size: Tuple of (height, width) for the image resolution

        Returns:
            map_x, map_y: Lookup maps for cv2.remap() to undistort images
            roi: Region of interest after undistortion (x, y, w, h)
        """
        h, w = image_size

        # Compute optimal camera matrix to handle black borders
        # alpha=0: crop all black pixels; alpha=1: keep all original pixels
        self.optimal_K, roi = cv2.getOptimalNewCameraMatrix(
            self.initial_K, self.initial_D, (w, h), alpha=0, newImgSize=(w, h)
        )

        # Generate lookup tables for fast image undistortion
        # CV_16SC2 format is faster and more memory-efficient than CV_32FC1
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.initial_K,
            self.initial_D,
            None,  # R (Rotation matrix) - None for monocular cameras
            self.optimal_K,  # New camera matrix with optimal parameters
            (w, h),
            cv2.CV_16SC2,
        )

        return map_x, map_y, roi, self.optimal_K

    def _track_features_bidirectional(self, img0, img1, p0):
        lk_params = dict(
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
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
            & (dist < self.config.fb_max_dist)
        )

        return p1, good_mask
