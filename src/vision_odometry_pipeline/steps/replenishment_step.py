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

        self._mask: np.ndarray | None = None
        self._mask_shape: tuple[int, int] | None = None

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
        if total_active > self.config.max_features * self.config.min_feature_factor:
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
        grid_counts = np.zeros(n_bins, dtype=np.int32)

        P = state.P
        C = state.C
        nP = len(P)
        nC = len(C)
        total_current = nP + nC

        def _accumulate_bins(pts: np.ndarray):
            if len(pts) == 0:
                return
            cx = (pts[:, 0] // dx).astype(np.int32)
            cy = (pts[:, 1] // dy).astype(np.int32)
            cx = np.clip(cx, 0, n_cols - 1)
            cy = np.clip(cy, 0, n_rows - 1)
            bin_ids = cy * n_cols + cx
            np.add.at(grid_counts, bin_ids, 1)

        _accumulate_bins(P)
        _accumulate_bins(C)

        # Reuse mask buffer (allocate once per resolution)
        if self._mask is None or self._mask_shape != curr_img.shape:
            self._mask = np.empty(curr_img.shape, dtype=np.uint8)
            self._mask_shape = curr_img.shape
        mask = self._mask
        mask.fill(255)

        if nP > 0:
            for pt in P:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), self.config.mask_radius, 0, -1)
        if nC > 0:
            for pt in C:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), self.config.mask_radius, 0, -1)

        n_needed = self.config.max_features - total_current
        if n_needed <= 0:
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

        keypoints = np.empty((0, 2), dtype=np.float32)

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
        if total_current + len(keypoints) > self.config.max_features:
            needed = self.config.max_features - total_current
            keypoints = keypoints[:needed] if needed > 0 else np.empty((0, 2))

        # Data Stacking (min allocations, no tile, no flatten copy)
        n_new = len(keypoints)
        if n_new > 0:
            n_old = len(state.C)

            full_C = np.empty((n_old + n_new, 2), dtype=np.float32)
            full_F = np.empty((n_old + n_new, 2), dtype=np.float32)
            full_T = np.empty((n_old + n_new, 12), dtype=np.float32)

            if n_old > 0:
                full_C[:n_old] = state.C
                full_F[:n_old] = state.F
                full_T[:n_old] = state.T_first

            full_C[n_old:] = keypoints
            full_F[n_old:] = keypoints

            pose_3x4 = state.pose[:3, :].ravel()  # prefer ravel over flatten (avoids copy when possible)
            pose_3x4 = pose_3x4.astype(np.float32, copy=False)

            # Broadcast to all new rows (no np.tile allocation)
            full_T[n_old:] = pose_3x4
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
