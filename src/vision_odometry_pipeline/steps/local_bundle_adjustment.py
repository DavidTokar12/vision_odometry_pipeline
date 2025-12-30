from __future__ import annotations

from collections import defaultdict
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from vision_odometry_pipeline.vo_state import VoState
from vision_odometry_pipeline.vo_step import VoStep


@dataclass(frozen=True)
class WindowFrame:
    """
    Represents a single frame snapshot within the sliding window.
    Stores immutable copies of the tracking state needed for optimization.
    """

    frame_id: int
    pose: np.ndarray  # T_WC (4x4)
    P: np.ndarray  # 2D Keypoints (N, 2)
    landmark_ids: np.ndarray  # Landmark IDs (N,)


class LocalBundleAdjustmentStep(VoStep):
    def __init__(self, K: np.ndarray, window_size: int = 10):
        super().__init__("LocalBundleAdjustment")
        self.K = K
        self.window_size = window_size

        # The sliding window buffer (deque efficiently handles pushes/pops)
        self._window: deque[WindowFrame] = deque(maxlen=window_size)

        # Cache for optimized poses: frame_id -> T_WC (4x4)
        # This acts as the "marginalization" prior. When a frame moves from
        # position 't' to 't-1' in the window, we want to remember its
        # refined pose from the previous optimization, not its initial PnP guess.
        self._optimized_poses: dict[int, np.ndarray] = {}

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Updates the sliding window.

        Returns:
            (Optimized_Pose, P, Optimized_X, Vis)
        """
        # 1. Manage Sliding Window
        self._update_window(state)

        # Minimum window size check (need at least 3 frames for meaningful BA)
        if len(self._window) < 3:
            return state.pose, state.P, state.X, None

        # 2. Filtering & Association
        # valid_obs: Dict[landmark_id, List[(frame_idx, keypoint_uv)]]
        # valid_3d:  Dict[landmark_id, Point_XYZ]
        valid_obs, valid_3d = self._prepare_optimization_data(state)

        # If not enough constraints, skip optimization
        if len(valid_3d) < 10:
            return state.pose, state.P, state.X, None

        # A. Pack Parameters
        x0, pose_indices, landmark_indices = self._pack_parameters(valid_obs, valid_3d)

        # B. Compute Sparsity Pattern
        sparsity = self._compute_sparsity(
            len(x0), pose_indices, landmark_indices, valid_obs
        )

        # C. Run Optimization
        # We need the anchor pose (Frame 0) to remain fixed
        anchor_frame = self._window[0]

        # Ensure the anchor uses the BEST known pose (refined in previous steps),
        # not the stale PnP guess from the snapshot.
        anchor_pose = self._optimized_poses.get(
            anchor_frame.frame_id, anchor_frame.pose
        )

        res = least_squares(
            fun=self._calculate_residuals,
            x0=x0,
            jac_sparsity=sparsity,
            verbose=0,
            x_scale="jac",
            ftol=1e-3,  # Loose tolerance for speed (real-time requirement)
            method="trf",  # Trust Region Reflective
            loss="huber",  # Robust loss function to ignore outliers
            f_scale=1.5,  # Outlier threshold in pixels
            args=(
                pose_indices,
                landmark_indices,
                valid_obs,
                anchor_pose,
            ),
        )

        # D. Unpack and Update State
        new_pose, new_X = self._unpack_results(
            res.x, state, pose_indices, landmark_indices
        )

        return new_pose, state.P, new_X, None

    def _unpack_results(
        self,
        x_opt: np.ndarray,
        state: VoState,
        pose_indices: dict[int, int],
        landmark_indices: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 5b: Unpack optimized parameters back into the State and History.
        """

        # 1. Update Optimized Poses in History Cache
        # (This ensures the next window starts with better guesses)
        for frame_idx, start_idx in pose_indices.items():
            # Get the frame ID from the window index
            frame_id = self._window[frame_idx].frame_id

            # Extract Vector
            pose_params = x_opt[start_idx : start_idx + 6]
            r_vec = pose_params[:3]
            t_vec = pose_params[3:]

            # Convert T_CW (optimized) back to T_WC (state format)
            R_CW, _ = cv2.Rodrigues(r_vec)

            # T_WC = (T_CW)^-1
            R_WC = R_CW.T
            t_WC = -R_WC @ t_vec.reshape(3, 1)

            T_WC = np.eye(4)
            T_WC[:3, :3] = R_WC
            T_WC[:3, 3] = t_WC.flatten()

            self._optimized_poses[frame_id] = T_WC

        # 2. Get the Optimized Pose for the CURRENT frame
        # The current frame is always the last one in the window
        curr_win_idx = len(self._window) - 1

        if curr_win_idx in pose_indices:
            # We have an optimized version
            curr_frame_id = self._window[curr_win_idx].frame_id
            new_pose = self._optimized_poses[curr_frame_id]
        else:
            # Fallback (shouldn't happen if logic is correct)
            new_pose = state.pose

        # 3. Update Landmarks (state.X)
        new_X = state.X.copy()

        # Map: Landmark ID -> Index in state.X array
        # We need this lookup to know WHICH row of new_X to update
        id_to_idx = {lid: i for i, lid in enumerate(state.landmark_ids)}

        for lid, start_idx in landmark_indices.items():
            if lid in id_to_idx:
                # Retrieve optimized 3D point
                pt_opt = x_opt[start_idx : start_idx + 3]

                # Update the global map at the correct index
                row_idx = id_to_idx[lid]
                new_X[row_idx] = pt_opt

        return new_pose, new_X

    def _compute_sparsity(
        self,
        num_params: int,
        pose_indices: dict[int, int],
        landmark_indices: dict[int, int],
        valid_observations: dict,
    ):
        """
        Step 4: Compute the Jacobian Sparsity Matrix.

        This defines which parameters affect which residuals.
        1 means "Non-Zero Derivative" (Parameter affects Residual).
        0 means "Zero Derivative" (Parameter does not affect Residual).
        """
        # Calculate total number of residuals (2 per observation: u and v)
        total_observations = sum(len(obs) for obs in valid_observations.values())
        num_residuals = total_observations * 2

        # LIL format is efficient for constructing sparse matrices incrementally
        sparsity = lil_matrix((num_residuals, num_params), dtype=int)

        row_idx = 0

        # IMPORTANT: Iterate in the EXACT same order as _calculate_residuals
        # Order: Landmark -> Frame -> (u, v)

        for lid, obs_list in valid_observations.items():
            # Get the column indices for this landmark's parameters (3 columns)
            lm_start_col = landmark_indices[lid]
            lm_cols = [lm_start_col, lm_start_col + 1, lm_start_col + 2]

            for frame_idx, _ in obs_list:
                # --- Fill Landmark Columns ---
                # The landmark position affects both u (row_idx) and v (row_idx+1)
                sparsity[row_idx, lm_cols] = 1
                sparsity[row_idx + 1, lm_cols] = 1

                # --- Fill Pose Columns ---
                # If frame is not the fixed anchor (0), its pose params affect residuals
                if frame_idx in pose_indices:
                    pose_start_col = pose_indices[frame_idx]
                    # 6 pose parameters
                    pose_cols = list(range(pose_start_col, pose_start_col + 6))

                    sparsity[row_idx, pose_cols] = 1
                    sparsity[row_idx + 1, pose_cols] = 1

                # Move to next pair of residuals
                row_idx += 2

        return sparsity

    def _calculate_residuals(
        self,
        x: np.ndarray,
        pose_indices: dict[int, int],
        landmark_indices: dict[int, int],
        valid_observations: dict,
        fixed_anchor_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Step 3: The Cost Function.
        Calculates the difference between projected and observed pixels.

        Args:
            x: Flattened optimization vector.
            fixed_anchor_pose: T_WC of the oldest frame (Frame 0), which is NOT optimized.
        """
        # 1. Unpack Camera Intrinsics
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # 2. Pre-compute the Fixed Anchor Extrinsics (T_CW)
        # We do this once because Frame 0 never changes during optimization.
        R_anchor_WC = fixed_anchor_pose[:3, :3]
        t_anchor_WC = fixed_anchor_pose[:3, 3]

        # Invert to T_CW
        R_anchor_CW = R_anchor_WC.T
        t_anchor_CW = -R_anchor_CW @ t_anchor_WC

        residuals = []

        # 3. Iterate over every valid constraint
        # valid_observations is Dict[landmark_id, List[(frame_idx, obs_uv)]]
        for lid, obs_list in valid_observations.items():
            # A. Retrieve Landmark 3D Position
            idx_pt = landmark_indices[lid]
            point_3d = x[idx_pt : idx_pt + 3]  # (3,)

            # B. Project into every observing frame
            for frame_idx, observed_uv in obs_list:
                # --- Get Camera Extrinsics (R, t) ---
                if frame_idx == 0:
                    # Use the Fixed Anchor
                    R, t = R_anchor_CW, t_anchor_CW
                else:
                    # Retrieve from optimization vector 'x'
                    idx_pose = pose_indices[frame_idx]
                    pose_params = x[idx_pose : idx_pose + 6]

                    r_vec = pose_params[:3]
                    t_vec = pose_params[3:]

                    # Convert Angle-Axis to Rotation Matrix
                    R, _ = cv2.Rodrigues(r_vec)
                    t = t_vec

                # --- Projection Logic (Pinhole Model) ---
                # X_cam = R * X_world + t
                X_cam = R @ point_3d + t

                # Check for invalid depth (behind camera) to avoid division by zero
                # We penalize this heavily or just let the optimizer handle the large error
                if X_cam[2] < 1e-4:
                    residuals.append(100.0)
                    residuals.append(100.0)
                    continue

                # Project to Pixel
                inv_z = 1.0 / X_cam[2]
                u_proj = fx * X_cam[0] * inv_z + cx
                v_proj = fy * X_cam[1] * inv_z + cy

                # --- Compute Error ---
                res_u = u_proj - observed_uv[0]
                res_v = v_proj - observed_uv[1]

                residuals.append(res_u)
                residuals.append(res_v)

        return np.array(residuals)

    def _pack_parameters(
        self,
        valid_observations: dict,
        valid_3d_points: dict,
    ) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
        """
        Step 2: Vectorize Poses and Landmarks into a single 1D array 'x0'.

        We optimize frames [1..N]. Frame 0 is the fixed anchor.
        We optimize World-to-Camera (T_CW) vectors (r, t) for direct projection.
        """
        x0_parts = []
        pose_indices = {}  # Map: frame_idx_in_window -> start_index_in_x0
        landmark_indices = {}  # Map: landmark_id -> start_index_in_x0
        current_idx = 0

        # --- A. Pack Poses (Skip Frame 0) ---
        for i in range(1, len(self._window)):
            frame = self._window[i]

            # --- FIX: Use the latest optimized pose if available ---
            # This ensures we start the solver from the best known state,
            # not the noisy initial PnP state.
            T_WC = self._optimized_poses.get(frame.frame_id, frame.pose)
            # -------------------------------------------------------

            # Convert T_WC (Pose) -> T_CW (Extrinsics)
            # T_CW = T_WC^-1
            R_WC = T_WC[:3, :3]
            t_WC = T_WC[:3, 3]

            R_CW = R_WC.T
            t_CW = -R_CW @ t_WC

            # Convert Matrix to Vector (Rodrigues)
            r_vec, _ = cv2.Rodrigues(R_CW)  # (3, 1)
            t_vec = t_CW  # (3,)

            # Flatten to 6 parameters [r1, r2, r3, t1, t2, t3]
            pose_params = np.hstack((r_vec.flatten(), t_vec.flatten()))
            x0_parts.append(pose_params)

            pose_indices[i] = current_idx
            current_idx += 6

        # --- B. Pack Landmarks ---
        # Sort IDs to ensure deterministic ordering in the vector
        sorted_lids = sorted(valid_3d_points.keys())

        for lid in sorted_lids:
            point_3d = valid_3d_points[lid]  # (3,)
            x0_parts.append(point_3d)

            landmark_indices[lid] = current_idx
            current_idx += 3

        x0 = np.hstack(x0_parts)
        return x0, pose_indices, landmark_indices

    def _prepare_optimization_data(self, state: VoState):
        """
        Step 1: Identify valid constraints.

        Criteria:
        1. Landmark must be currently active (exist in state.X).
        2. Landmark must be observed in at least 2 frames within the window.
        """
        # A. Create Lookup for Current 3D Positions (ID -> XYZ)
        # We only optimize landmarks that are currently tracked (in state.X).
        # Lost landmarks are ignored as we don't store their 3D positions in history.
        curr_map_3d = {lid: x for lid, x in zip(state.landmark_ids, state.X)}

        # B. Collect Observations across the Window
        # Map: landmark_id -> list of (frame_index_in_window, (u, v))
        observations = defaultdict(list)

        for frame_idx, frame in enumerate(self._window):
            # Iterate through all keypoints in this frame snapshot
            # frame.P and frame.landmark_ids are aligned arrays
            for i, lid in enumerate(frame.landmark_ids):
                # Optimization: Only collect if it's a known active landmark
                if lid in curr_map_3d:
                    observations[lid].append((frame_idx, frame.P[i]))

        # C. Filter: Keep only landmarks with >= 2 observations (Parallax requirement)
        valid_observations = {}
        valid_3d_points = {}

        for lid, obs_list in observations.items():
            if len(obs_list) >= 2:
                valid_observations[lid] = obs_list
                valid_3d_points[lid] = curr_map_3d[lid]

        return valid_observations, valid_3d_points

    def _update_window(self, state: VoState):
        """
        Ingests the current state into the sliding window.
        """
        # Check if we have a better (optimized) estimate for the current frame
        # (e.g., if we were doing loop closure), otherwise use the current state pose.
        current_pose = self._optimized_poses.get(state.frame_id, state.pose.copy())

        # Create Snapshot
        # Critical: We must .copy() numpy arrays because the pipeline reuses
        # buffers or replaces them in future steps.
        frame_snapshot = WindowFrame(
            frame_id=state.frame_id,
            pose=current_pose,
            P=state.P.copy(),
            landmark_ids=state.landmark_ids.copy(),
        )

        # Handle Cleanup BEFORE appending (because maxlen will auto-pop)
        if len(self._window) == self.window_size:
            # The oldest frame (index 0) is about to fall off the edge
            removed_frame = self._window[0]

            # Remove from optimization cache to prevent memory leaks
            if removed_frame.frame_id in self._optimized_poses:
                del self._optimized_poses[removed_frame.frame_id]

        # Add to window (auto-pops oldest if full)
        self._window.append(frame_snapshot)
