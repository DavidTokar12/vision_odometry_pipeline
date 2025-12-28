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

        # 1. Masking
        mask = np.full(curr_img.shape, 255, dtype=np.uint8)
        all_pts = np.vstack([state.P, state.C]) if len(state.C) > 0 else state.P
        if len(all_pts) > 0:
            for pt in all_pts:
                cv2.circle(
                    mask, (int(pt[0]), int(pt[1])), self.config.mask_radius, 0, -1
                )

        # 2. Detection
        h, w = curr_img.shape
        # Calculate target features per cell
        n_cells = self.config.grid_rows * self.config.grid_cols
        features_per_cell = int(self.config.max_features / n_cells)

        w_step = w // self.config.grid_cols
        h_step = h // self.config.grid_rows

        new_keypoints_list = []

        for r in range(self.config.grid_rows):
            for c in range(self.config.grid_cols):
                # Define ROI limits
                x_start, x_end = c * w_step, (c + 1) * w_step
                y_start, y_end = r * h_step, (r + 1) * h_step

                # Adjust last row/col to cover remainder
                if c == self.config.grid_cols - 1:
                    x_end = w
                if r == self.config.grid_rows - 1:
                    y_end = h

                # Count existing points in this cell
                in_region = (
                    (
                        (all_pts[:, 0] >= x_start)
                        & (all_pts[:, 0] < x_end)
                        & (all_pts[:, 1] >= y_start)
                        & (all_pts[:, 1] < y_end)
                    )
                    if len(all_pts) > 0
                    else []
                )

                n_existing = np.sum(in_region)
                n_needed_cell = features_per_cell - n_existing

                if n_needed_cell > 0:
                    # Extract ROI from image and mask
                    img_roi = curr_img[y_start:y_end, x_start:x_end]
                    mask_roi = mask[y_start:y_end, x_start:x_end]

                    pts = cv2.goodFeaturesToTrack(
                        img_roi,
                        mask=mask_roi,
                        maxCorners=n_needed_cell,
                        qualityLevel=self.config.quality_level,
                        minDistance=self.config.min_dist,
                        blockSize=self.config.block_size,
                        useHarrisDetector=self.config.use_harris,
                        k=self.config.harris_k,
                    )

                    if pts is not None:
                        # Additional Harris response filtering for even more selectivity
                        if self.config.use_harris and self.config.harris_threshold > 0:
                            pts = self._filter_by_harris_response(pts, img_roi)
                        else:
                            pts = pts.reshape(-1, 2)

                        if len(pts) > 0:
                            # Convert to global coordinates
                            pts[:, 0] += x_start
                            pts[:, 1] += y_start
                            new_keypoints_list.append(pts)

        if new_keypoints_list:
            keypoints = np.vstack(new_keypoints_list)
        else:
            keypoints = np.empty((0, 2), dtype=np.float32)

        # 3. Data Stacking
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

    def _filter_by_harris_response(
        self, pts: np.ndarray, img_roi: np.ndarray
    ) -> np.ndarray:
        pts = pts.reshape(-1, 2)
        # Compute Harris response for each point
        harris_responses = []
        for pt in pts:
            x_local = int(pt[0])
            y_local = int(pt[1])
            # Extract small window around point
            window_size = self.config.block_size
            half_w = window_size // 2
            y1 = max(0, y_local - half_w)
            y2 = min(img_roi.shape[0], y_local + half_w + 1)
            x1 = max(0, x_local - half_w)
            x2 = min(img_roi.shape[1], x_local + half_w + 1)

            if y2 > y1 and x2 > x1:
                window = img_roi[y1:y2, x1:x2]
                # Compute Harris response
                Ix = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
                Iy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
                Ixx = np.mean(Ix * Ix)
                Iyy = np.mean(Iy * Iy)
                Ixy = np.mean(Ix * Iy)

                det = Ixx * Iyy - Ixy * Ixy
                trace = Ixx + Iyy
                response = det - self.config.harris_k * (trace**2)
                harris_responses.append(response)
            else:
                harris_responses.append(0.0)

        # Normalize responses and filter
        if len(harris_responses) > 0:
            max_response = max(harris_responses)
            if max_response > 0:
                normalized_responses = np.array(harris_responses) / max_response
                valid_mask = normalized_responses > self.config.harris_threshold
                return pts[valid_mask]

        return pts

    def _visualize_new_features(self, img, mask, new_pts):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis[mask == 0] = vis[mask == 0] // 2
        for pt in new_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
        return vis
