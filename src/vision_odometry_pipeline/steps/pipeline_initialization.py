from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_configs import InitializationConfig
from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class PipelineInitialization(VoStep):
    def __init__(self, config: InitializationConfig, K, D) -> None:
        super().__init__("PipelineInitialization")
        self.config = config
        self.initial_K = K
        self.initial_D = D
        self.optimal_K = None

        # For debug using the use_gt_init option
        self._cached_gt_poses = None

    def process(self, state: VoState, debug: bool):
        img_prev = state.image_buffer.prev
        img_curr = state.image_buffer.curr

        # --- Tracking ---

        p0 = state.C
        p1, st = self._track_features_bidirectional(img_prev, img_curr, p0)
        p1 = p1.reshape(-1, 2)

        # Update Candidates (Keep only tracked points)
        st = st.flatten() == 1
        new_C = p1[st]  # Current pixel coords
        new_F = state.F[st]  # First pixel coords (Init start)
        new_T = state.T_first[st]

        if len(new_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, None, None, False

        # For debug using the use_gt_init option
        if self.config.use_gt_init:
            return self._process_with_gt(state, new_C, new_F, new_T)

        # --- Bucketing ---

        h, w = img_curr.shape
        grid_rows = self.config.grid_rows
        grid_cols = self.config.grid_cols
        grid_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)

        # Vectorized bucket index calculation
        # Normalized coordinates 0.0 to 1.0
        norm_x = new_C[:, 0] / w
        norm_y = new_C[:, 1] / h

        # Map to grid indices (0 to 9)
        idx_x = np.floor(norm_x * grid_cols).astype(np.int32)
        idx_y = np.floor(norm_y * grid_rows).astype(np.int32)

        # Clip to ensure valid indices (in case a point is slightly outside)
        idx_x = np.clip(idx_x, 0, grid_cols - 1)
        idx_y = np.clip(idx_y, 0, grid_rows - 1)

        # Count occupancy
        bin_ids = idx_y * grid_cols + idx_x
        np.add.at(grid_counts.ravel(), bin_ids, 1)

        occupied_cells = np.sum(grid_counts > 0)

        if occupied_cells < self.config.min_grid_occupancy:
            if debug:
                print(
                    f"[Init] Poor distribution. Occupied cells: {occupied_cells} < {self.config.min_grid_occupancy}"
                )
            return new_C, new_F, new_T, None, None, None, None, None, False

        # --- Baseline Check (Parallax Angle) ---

        # Reconstruct bearing vectors (assuming undistorted images & optimal_K)
        fx, fy = self.optimal_K[0, 0], self.optimal_K[1, 1]
        cx, cy = self.optimal_K[0, 2], self.optimal_K[1, 2]

        # Convert to normalized coordinates: (u - cx) / fx
        vec_C = np.stack(
            [
                (new_C[:, 0] - cx) / fx,
                (new_C[:, 1] - cy) / fy,
                np.ones(len(new_C), dtype=np.float32),
            ],
            axis=1,
            dtype=np.float32,
        )

        vec_F = np.stack(
            [
                (new_F[:, 0] - cx) / fx,
                (new_F[:, 1] - cy) / fy,
                np.ones(len(new_F), dtype=np.float32),
            ],
            axis=1,
            dtype=np.float32,
        )

        # Normalize to unit vectors
        vec_C /= np.linalg.norm(vec_C, axis=1, keepdims=True)
        vec_F /= np.linalg.norm(vec_F, axis=1, keepdims=True)

        # Compute angle (Dot Product)
        cos_angles = np.clip(np.sum(vec_C * vec_F, axis=1), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_angles))

        # Check global baseline condition (Median Angle)
        median_angle = np.median(angles_deg)
        if median_angle < self.config.min_parallax_angle:
            if debug:
                print(f"[Init] Low parallax. Median: {median_angle:.2f} deg")
            return new_C, new_F, new_T, None, None, None, None, None, False

        # Create selection mask: Filter points with very low individual parallax
        # (Using half the global threshold to keep points that are contributing)
        ready_mask = angles_deg > (
            self.config.min_parallax_angle * self.config.parallax_factor
        )

        # --- Compute Essential Matrx ---

        # Only use the 'ready' points to compute the Essential Matrix
        cand_C = new_C[ready_mask]

        # Only use the 'ready' points to compute the Essential Matrix
        cand_C = new_C[ready_mask]
        cand_F = new_F[ready_mask]

        # Compute Essential Matrix
        E, mask_ess = cv2.findEssentialMat(
            cand_F,
            cand_C,
            self.optimal_K,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold,
        )

        if E is None:
            return new_C, new_F, new_T, None, None, None, None, None, False

        # Filter by Essential Matrix Inliers
        mask_ess = mask_ess.ravel() == 1
        cand_C = cand_C[mask_ess]
        cand_F = cand_F[mask_ess]

        if len(cand_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, None, None, False

        # --- Pose Recovery and Cheirality check

        _, R, t, mask_pose = cv2.recoverPose(E, cand_F, cand_C, self.optimal_K)
        R, t = R.astype(np.float32), t.astype(np.float32)

        # Scale decision
        t = t / np.linalg.norm(t)

        # Filter by Cheirality
        pose_inliers = mask_pose.ravel() > 0
        cand_C = cand_C[pose_inliers]
        cand_F = cand_F[pose_inliers]

        if len(cand_C) < self.config.min_inliers:
            return new_C, new_F, new_T, None, None, None, None, None, False

        # --- Triangulation ---

        M0 = self.optimal_K @ np.hstack(
            (np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32))
        )
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

            new_ids = np.arange(len(new_X), dtype=np.int64)

            new_pose = np.eye(4, dtype=np.float32)
            new_pose[:3, :3] = R
            new_pose[:3, 3] = t.flatten()

            # Calculate Initial Average Depth
            # Transform points to Camera Frame to measure depth
            # X_cam = R * X_w + t
            X_cam = (R @ new_X.T).T + t.flatten()
            depths = np.linalg.norm(X_cam, axis=1)
            avg_depth = np.mean(depths)

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

            return rem_C, rem_F, rem_T, new_X, new_ids, new_P, new_pose, avg_depth, True

        return new_C, new_F, new_T, None, None, None, None, None, False

    def create_undistorted_maps(self, image_size):
        """
        Generate lookup maps to remove image distortion.
        """
        h, w = image_size

        if self.initial_D is None or np.all(np.abs(self.initial_D) < 1e-5):
            print("[Init] Rectified image detected (D~0). Forcing Original K.")
            self.optimal_K = self.initial_K.copy()

            # Update internal params
            self.fx, self.fy = self.optimal_K[0, 0], self.optimal_K[1, 1]
            self.cx, self.cy = self.optimal_K[0, 2], self.optimal_K[1, 2]

            # Create Identity Maps (No remapping/interpolation)
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)
            roi = (0, 0, w, h)

            return map_x, map_y, roi, self.optimal_K

        self.optimal_K, roi = cv2.getOptimalNewCameraMatrix(
            self.initial_K, self.initial_D, (w, h), alpha=0, newImgSize=(w, h)
        )

        map_x, map_y = cv2.initUndistortRectifyMap(
            self.initial_K,
            self.initial_D,
            None,
            self.optimal_K,
            (w, h),
            cv2.CV_16SC2,
        )

        # Adjust Principal Point for the ROI crop
        x, y, _, _ = roi
        self.optimal_K[0, 2] -= x  # cx' = cx - x
        self.optimal_K[1, 2] -= y  # cy' = cy - y

        # Update focal lengths and principal point with optimal values
        self.fx, self.fy = self.optimal_K[0, 0], self.optimal_K[1, 1]
        self.cx, self.cy = self.optimal_K[0, 2], self.optimal_K[1, 2]

        return map_x, map_y, roi, self.optimal_K

    def _track_features_bidirectional(self, img0, img1, p0):
        lk_params = {
            "winSize": self.config.lk_win_size,
            "maxLevel": self.config.lk_max_level,
            "criteria": self.config.lk_criteria,
        }
        # Forward flow
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # Backward flow
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        # Check consistency (L-infinity norm)
        dist = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good_mask = (
            (st1.flatten() == 1)
            & (st2.flatten() == 1)
            & (dist < self.config.fb_max_dist)
        )

        return p1, good_mask

    def find_initial_features(self, state: VoState):
        img = state.image_buffer.curr

        # Split image into a 4x4 grid (16 tiles) to force distribution
        # You can adjust grid_size based on resolution (e.g., 4 or 5)
        h, w = img.shape
        h_step = h // self.config.tile_rows
        w_step = w // self.config.tile_cols

        # Lower contrast threshold slightly to find points in duller areas
        # nfeatures per tile cap prevents one tile from dominating
        sift = cv2.SIFT_create(
            nfeatures=self.config.n_features,
            contrastThreshold=self.config.contrast_threshold,
        )

        all_keypoints = []

        for r in range(self.config.grid_rows):
            for c in range(self.config.grid_cols):
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

        if len(keypoints) < self.config.min_init_features:
            print("Warning: Low feature count in initialization frame")

        identity_pose_flat = np.hstack(
            (np.eye(3), np.zeros((3, 1))), dtype=np.float32
        ).flatten()
        T_first_init = np.tile(identity_pose_flat, (len(keypoints), 1))

        return keypoints, keypoints, T_first_init

    # For debug using the use_gt_init option
    def _process_with_gt(self, state, cand_C, cand_F, cand_T):
        """Hijacked logic using Ground Truth."""
        if self._cached_gt_poses is None:
            self._cached_gt_poses = self._load_poses(self.config.DEBUG_GT_POSES_PATH)

        # Get Poses for Frame 0 (Start) and Current Frame
        # Assuming init started at frame 0. If your buffer started later, adjust index.

        prev_idx = 0
        curr_idx = 1

        assert state.frame_id == 1

        try:
            T_wc_0 = self._cached_gt_poses[prev_idx]
            T_wc_1 = self._cached_gt_poses[curr_idx]
        except IndexError:
            print(
                f"[Init-GT] Error: GT Poses not found for frames {prev_idx}-{curr_idx}"
            )
            return cand_C, cand_F, cand_T, None, None, None, None, None, False

        print(f"[Init-GT] Cheating with GT Poses for Frame {prev_idx} -> {curr_idx}")

        # Invert to T_CW (World-to-Camera) for Projection Matrices
        T_cw_0 = self._inv_pose(T_wc_0)
        T_cw_1 = self._inv_pose(T_wc_1)

        P0 = self.optimal_K @ T_cw_0[:3, :]
        P1 = self.optimal_K @ T_cw_1[:3, :]

        # Triangulate
        pts4D = cv2.triangulatePoints(P0, P1, cand_F.T, cand_C.T)

        # Filter
        pts_hom = pts4D[:3]
        W = pts4D[3]
        mask_valid = np.abs(W) > 1e-4

        # Check Depth (Cheirality)
        if np.any(mask_valid):
            points3D = pts_hom[:, mask_valid] / W[mask_valid]  # In World Frame

            # Check depth in current camera
            X_cam = (T_cw_1[:3, :3] @ points3D) + T_cw_1[:3, 3][:, None]
            in_front = X_cam[2] > 0
            mask_valid[mask_valid] = in_front

        num_valid = np.sum(mask_valid)

        if num_valid > self.config.min_inliers:
            print(f"[Init-GT] Success! {num_valid} landmarks initialized via GT.")

            new_X = (pts_hom[:, mask_valid] / W[mask_valid]).T.astype(np.float32)
            new_P = cand_C[mask_valid]

            new_ids = np.arange(len(new_X), dtype=np.int64)

            # Return the GT Pose (T_WC) as the initial pose state
            new_pose = np.eye(4, dtype=np.float32)
            new_pose[:3, :] = T_wc_1[:3, :]  # Use T_WC
            rem_C = cand_C[~mask_valid]
            rem_F = cand_F[~mask_valid]
            rem_T = cand_T[~mask_valid].copy()

            gt_start_pose_flat = T_wc_0[:3, :].flatten().astype(np.float32)
            rem_T[:] = gt_start_pose_flat

            return rem_C, rem_F, rem_T, new_X, new_ids, new_P, new_pose, 1, True

        print(f"[Init-GT] Waiting for baseline... ({num_valid} valid points)")
        return cand_C, cand_F, cand_T, None, None, None, None, None, False

    def _load_poses(self, path):
        poses = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                vals = [float(x) for x in line.split()]
                poses.append(np.array(vals, dtype=np.float32).reshape(3, 4))
        return poses

    def _inv_pose(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        return np.hstack((R.T, (-R.T @ t).reshape(3, 1)))
