from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


# --- Configuration ---
@dataclass
class InitializationConfig:
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 5
    fb_max_dist: float = 1.0
    ransac_threshold: float = 0.5
    ransac_prob: float = 0.999
    min_buffer_size: int = 2
    min_inliers: int = 100
    min_parallax_angle: float = 2.0  # Minimum median angle in degrees
    min_grid_occupancy: int = 15
    # find_initial_features() still used hardcoded parameters


class PipelineInitialization(VoStep):
    def __init__(self, K, D) -> None:
        super().__init__("PipelineInitialization")
        self.config = InitializationConfig()
        self.initial_K = K
        self.initial_D = D
        self.optimal_K = None

    def process(self, state: VoState, debug: bool):
        img_prev = state.image_buffer.prev
        img_curr = state.image_buffer.curr

        # 1. Track Keypoints
        p0 = state.C.astype(np.float32)
        p1, st = self._track_features_bidirectional(img_prev, img_curr, p0)
        p1 = p1.reshape(-1, 2)

        # 2. Update Candidates (Keep only tracked points)
        st = st.flatten() == 1
        new_C = p1[st]  # Current pixel coords
        new_F = state.F[st]  # First pixel coords (Init start)
        new_T = state.T_first[st]

        if len(new_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, False

        if len(new_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, False

        # --- NEW: Spatial Distribution Check (Bucketing) ---
        # Goal: Ensure features are well-distributed across the image
        h, w = img_curr.shape
        grid_rows, grid_cols = 10, 10  # 10x10 grid
        grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

        # Vectorized bucket index calculation
        # Normalized coordinates 0.0 to 1.0
        norm_x = new_C[:, 0] / w
        norm_y = new_C[:, 1] / h

        # Map to grid indices (0 to 9)
        idx_x = np.floor(norm_x * grid_cols).astype(int)
        idx_y = np.floor(norm_y * grid_rows).astype(int)

        # Clip to ensure valid indices (in case a point is slightly outside)
        idx_x = np.clip(idx_x, 0, grid_cols - 1)
        idx_y = np.clip(idx_y, 0, grid_rows - 1)

        # Count occupancy
        # (This loop is fast for ~200 points)
        for r, c in zip(idx_y, idx_x):
            grid_counts[r, c] += 1

        occupied_cells = np.sum(grid_counts > 0)

        if occupied_cells < self.config.min_grid_occupancy:
            if debug:
                print(
                    f"[Init] Poor distribution. Occupied cells: {occupied_cells} < {self.config.min_grid_occupancy}"
                )
            return new_C, new_F, new_T, None, None, None, False
        # ---------------------------------------------------

        # --- CHANGED: Baseline Check (Parallax Angle) ---
        # 1. Reconstruct bearing vectors (assuming undistorted images & optimal_K)
        fx, fy = self.optimal_K[0, 0], self.optimal_K[1, 1]
        cx, cy = self.optimal_K[0, 2], self.optimal_K[1, 2]

        # Convert to normalized coordinates: (u - cx) / fx
        vec_C = np.stack(
            [(new_C[:, 0] - cx) / fx, (new_C[:, 1] - cy) / fy, np.ones(len(new_C))],
            axis=1,
        )
        vec_F = np.stack(
            [(new_F[:, 0] - cx) / fx, (new_F[:, 1] - cy) / fy, np.ones(len(new_F))],
            axis=1,
        )

        # 2. Normalize to unit vectors
        vec_C /= np.linalg.norm(vec_C, axis=1, keepdims=True)
        vec_F /= np.linalg.norm(vec_F, axis=1, keepdims=True)

        # 3. Compute angle (Dot Product)
        # Clip to [-1, 1] to avoid NaN errors from floating point precision
        cos_angles = np.clip(np.sum(vec_C * vec_F, axis=1), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))

        # 4. Check global baseline condition (Median Angle)
        median_angle = np.median(angles_deg)
        if median_angle < self.config.min_parallax_angle:
            if debug:
                print(f"[Init] Low parallax. Median: {median_angle:.2f} deg")
            return new_C, new_F, new_T, None, None, None, False

        # Create selection mask: Filter points with very low individual parallax
        # (Using half the global threshold to keep points that are contributing)
        ready_mask = angles_deg > (self.config.min_parallax_angle * 0.5)
        # -------------------------------------------------

        # Only use the 'ready' points to compute the Essential Matrix
        cand_C = new_C[ready_mask]

        # Only use the 'ready' points to compute the Essential Matrix
        cand_C = new_C[ready_mask]
        cand_F = new_F[ready_mask]

        # 3. Compute Essential Matrix
        E, mask_ess = cv2.findEssentialMat(
            cand_F,
            cand_C,
            self.optimal_K,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold,
        )

        if E is None:
            return new_C, new_F, new_T, None, None, None, False

        # Filter by Essential Matrix Inliers
        mask_ess = mask_ess.ravel() == 1
        cand_C = cand_C[mask_ess]
        cand_F = cand_F[mask_ess]

        if len(cand_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, False

        # Recover Pose (R, t)
        _, R, t, mask_pose = cv2.recoverPose(E, cand_F, cand_C, self.optimal_K)

        # Scale decision
        t = t / np.linalg.norm(t)

        # Filter by Cheirality
        pose_inliers = mask_pose.ravel() > 0
        cand_C = cand_C[pose_inliers]
        cand_F = cand_F[pose_inliers]

        if len(cand_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, False

        # Triangulation
        M0 = self.optimal_K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        M1 = self.optimal_K @ np.hstack((R, t))

        points4D = cv2.triangulatePoints(M0, M1, cand_F.T, cand_C.T)

        pts_hom = points4D[:3]
        W = points4D[3]
        mask_finite = np.abs(W) > 1e-4
        valid_tri_mask = np.zeros(W.shape, dtype=bool)

        if np.any(mask_finite):
            points3D = pts_hom[:, mask_finite] / W[mask_finite]
            in_front_1 = points3D[2] > 0
            X_local = (R @ points3D) + t
            in_front_2 = X_local[2] > 0
            valid_tri_mask[mask_finite] = in_front_1 & in_front_2

        num_valid = np.sum(valid_tri_mask)
        print(f"[Init] {num_valid} landmarks detected.")

        if num_valid > self.config.min_inliers:
            print(f"[Init] Success! {num_valid} landmarks initialized.")

            new_X = (pts_hom[:, valid_tri_mask] / W[valid_tri_mask]).T
            new_P = cand_C[valid_tri_mask]

            new_pose = np.eye(4)
            new_pose[:3, :3] = R
            new_pose[:3, 3] = t.flatten()

            # Cleanup Candidates
            keep_mask = np.ones(len(new_C), dtype=bool)

            # Map masks back to original indices
            idx_ready = np.where(ready_mask)[0]
            idx_ess = idx_ready[mask_ess]
            idx_pose = idx_ess[pose_inliers]
            idx_final = idx_pose[valid_tri_mask]

            keep_mask[idx_final] = False

            rem_C = new_C[keep_mask]
            rem_F = new_F[keep_mask]
            rem_T = new_T[keep_mask]

            return rem_C, rem_F, rem_T, new_X, new_P, new_pose, True

        return new_C, new_F, new_T, None, None, None, False

    def find_initial_features(self, state: VoState):
        img = state.image_buffer.curr

        # Split image into a 4x4 grid (16 tiles) to force distribution
        # You can adjust grid_size based on resolution (e.g., 4 or 5)
        grid_rows, grid_cols = 4, 4
        h, w = img.shape
        h_step = h // grid_rows
        w_step = w // grid_cols

        # Lower contrast threshold slightly to find points in duller areas
        # nfeatures per tile cap prevents one tile from dominating
        sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.03)

        all_keypoints = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                # Define ROI (Region of Interest)
                x1, x2 = c * w_step, (c + 1) * w_step
                y1, y2 = r * h_step, (r + 1) * h_step

                # Detect in the tile
                tile = img[y1:y2, x1:x2]
                kps = sift.detect(tile, None)

                # Shift keypoint coordinates back to global frame
                for kp in kps:
                    kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                    all_keypoints.append(kp)

        keypoints = np.array([kp.pt for kp in all_keypoints], dtype=np.float32).reshape(
            -1, 2
        )

        if len(keypoints) < 15:
            print("Warning: Low feature count in initialization frame")

        identity_pose_flat = np.hstack((np.eye(3), np.zeros((3, 1)))).flatten()
        T_first_init = np.tile(identity_pose_flat, (len(keypoints), 1))

        return keypoints, keypoints, T_first_init

    def create_undistorted_maps(self, image_size):
        """
        Generate lookup maps to remove image distortion.
        """
        h, w = image_size
        self.optimal_K, roi = cv2.getOptimalNewCameraMatrix(
            self.initial_K, self.initial_D, (w, h), alpha=0, newImgSize=(w, h)
        )

        # Update focal lengths and principal point with optimal values
        self.fx, self.fy = self.optimal_K[0, 0], self.optimal_K[1, 1]
        self.cx, self.cy = self.optimal_K[0, 2], self.optimal_K[1, 2]

        map_x, map_y = cv2.initUndistortRectifyMap(
            self.initial_K,
            self.initial_D,
            None,
            self.optimal_K,
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
