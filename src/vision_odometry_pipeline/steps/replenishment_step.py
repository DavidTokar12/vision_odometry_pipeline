from __future__ import annotations

import cv2
import numpy as np

from vision_odometry_pipeline.vo_configs import ReplenishmentConfig
from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


class ReplenishmentStep(VoStep):
    def __init__(self):
        super().__init__("Replenishment")
        self.config = ReplenishmentConfig()

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Detects new corners and STACKS them onto the state arrays.

        Returns:
            (Full_C, Full_F, Full_T, Vis)
        """
        curr_img = state.image_buffer.curr
        h, w = curr_img.shape

        total_active = len(state.P) + len(state.C)
        if total_active > self.config.max_features * 0.6:
            return state.C, state.F, state.T_first, None

        # Grid Configuration
        n_rows = self.config.grid_rows
        n_cols = self.config.grid_cols
        dy = h // n_rows
        dx = w // n_cols
        n_bins = n_rows * n_cols

        # Define Caps: Allow 1.5x overflow to prioritize quality over strict uniformity
        avg_features_per_cell = self.config.max_features / n_bins
        cap_per_cell = int(avg_features_per_cell * self.config.cell_cap_multiplier)

        # Count Existing Features and map them to their grid bin ID
        grid_counts = np.zeros(n_bins, dtype=int)
        all_pts = np.vstack([state.P, state.C]) if len(state.C) > 0 else state.P

        if len(all_pts) > 0:
            cx = (all_pts[:, 0] // dx).astype(int)
            cy = (all_pts[:, 1] // dy).astype(int)
            cx = np.clip(cx, 0, n_cols - 1)
            cy = np.clip(cy, 0, n_rows - 1)
            bin_ids = cy * n_cols + cx
            np.add.at(grid_counts, bin_ids, 1)

        # Masking out already existing points
        mask = np.full(curr_img.shape, 255, dtype=np.uint8)
        if len(all_pts) > 0:
            for pt in all_pts:
                cv2.circle(
                    mask, (int(pt[0]), int(pt[1])), self.config.mask_radius, 0, -1
                )

        n_needed = self.config.max_features - len(all_pts)
        if not n_needed > 0:
            return state.C, state.F, state.T_first, None

        # Global Detection
        candidates = cv2.goodFeaturesToTrack(
            curr_img,
            mask=mask,
            maxCorners=self.config.max_features * self.config.global_feature_multiplier,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_dist,
            blockSize=self.config.block_size,
            useHarrisDetector=self.config.use_harris,
            k=self.config.harris_k,
        )

        if candidates is None:
            return state.C, state.F, state.T_first, None

        candidates = candidates.reshape(-1, 2)

        keypoints = np.empty((0, 2))

        # Map candidates to bins
        c_idx = (candidates[:, 0] // dx).astype(int)
        r_idx = (candidates[:, 1] // dy).astype(int)
        c_idx = np.clip(c_idx, 0, n_cols - 1)
        r_idx = np.clip(r_idx, 0, n_rows - 1)
        cand_bins = r_idx * n_cols + c_idx  # Vector holding bin IDs for every candidate

        # Sort by bin ID (Stable sort preserves quality order within bins)
        sort_idx = np.argsort(cand_bins, kind="stable")  # idx of cand's grouped by bin
        sorted_bins = cand_bins[sort_idx]  # candidates grouped by bin

        # Determine Rank within Bin
        # Find indices where the bin ID changes
        unique_bins, run_starts = np.unique(sorted_bins, return_index=True)
        ranks = np.zeros(len(candidates), dtype=int)

        for i, _ in enumerate(unique_bins):
            start = run_starts[i]
            end = run_starts[i + 1] if i + 1 < len(unique_bins) else len(candidates)
            count = end - start
            ranks[sort_idx[start:end]] = np.arange(count)

        # Calculate how many spots are left in each candidate's bin
        available_space = np.maximum(0, cap_per_cell - grid_counts)  # len: num. bins

        # Only keep points whose rank fits in the available space
        keep_mask = ranks < available_space[cand_bins]  # len: num. candidates
        keypoints = candidates[keep_mask]  # len: num. needed points

        # Truncate if we exceeded the hard global limit
        total_current = len(all_pts)
        if total_current + len(keypoints) > self.config.max_features:
            needed = self.config.max_features - total_current
            keypoints = keypoints[:needed] if needed > 0 else np.empty((0, 2))

        # Data Stacking
        if len(keypoints) > 0:
            full_C = np.vstack([state.C, keypoints])
            full_F = np.vstack([state.F, keypoints])  # F is current pixel loc

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
            # Draw Grid Lines
            for i in range(1, n_cols):
                cv2.line(vis, (i * dx, 0), (i * dx, h), (50, 50, 50), 1)
            for i in range(1, n_rows):
                cv2.line(vis, (0, i * dy), (w, i * dy), (50, 50, 50), 1)

        return full_C, full_F, full_T, vis

    def _visualize_new_features(self, img, mask, new_pts):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[mask == 0] = vis[mask == 0] // 2
        for pt in new_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
        return vis
