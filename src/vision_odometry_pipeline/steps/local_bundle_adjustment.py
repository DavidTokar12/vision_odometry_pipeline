from __future__ import annotations

from collections import defaultdict
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from vision_odometry_pipeline.vo_configs import LocalBundleAdjustmentConfig
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


def _batched_rodrigues(r_vecs: np.ndarray) -> np.ndarray:
    """
    Vectorized implementation of Rodrigues' rotation formula.
    Args:
        r_vecs: (N, 3) array of rotation vectors.
    Returns:
        (N, 3, 3) array of rotation matrices.
    """
    theta = np.linalg.norm(r_vecs, axis=1, keepdims=True)  # (N, 1)

    # Avoid division by zero for small angles
    # For theta ~ 0, R ~ I + [r]_x. We use a mask to handle this safely.
    with np.errstate(invalid="ignore", divide="ignore"):
        k = r_vecs / theta

    # Handle zero rotation vectors (theta is zero)
    # We replace NaNs in k with 0 (or any value) because they will be zeroed out by sin/lines anyway,
    # but strictly we should handle the identity case.
    # A cleaner numerical approach is to compute coefficients.

    # K = [k]_x skew symmetric matrix
    K = np.zeros((r_vecs.shape[0], 3, 3))
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    # Rodrigues formula: I + sin(theta)K + (1-cos(theta))K^2
    I_matrix = np.eye(3).reshape(1, 3, 3)

    # Fix NaNs for the zero-angle case (where theta=0 resulted in k=NaN)
    # If theta is close to 0, limit tends to Identity.
    # We apply the formula where theta > epsilon.

    # Re-calculate cleanly to ensure 1000% stability
    theta = theta.reshape(-1, 1, 1)
    k_cos = 1 - np.cos(theta)
    k_sin = np.sin(theta)

    # Where theta is 0, these terms result in 0 update to Identity, which is correct.
    # However, k has NaNs. We must zero out the update terms where theta is 0.
    mask = theta > 1e-8

    term1 = mask * k_sin * K
    term2 = mask * k_cos * (K @ K)

    return I_matrix + term1 + term2


def _batched_skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Computes skew-symmetric matrices for a batch of 3D vectors.
    Args:
        v: (N, 3) vectors
    Returns:
        (N, 3, 3) skew-symmetric matrices
    """
    N = v.shape[0]
    z = np.zeros(N)
    # [0, -z, y]
    # [z, 0, -x]
    # [-y, x, 0]

    # We construct the 3x3 manually
    # S[:, 0, 1] = -v[:, 2] etc...
    # Stacking is cleaner for numpy

    row0 = np.stack([z, -v[:, 2], v[:, 1]], axis=1)  # (N, 3)
    row1 = np.stack([v[:, 2], z, -v[:, 0]], axis=1)
    row2 = np.stack([-v[:, 1], v[:, 0], z], axis=1)

    return np.stack([row0, row1, row2], axis=1)  # (N, 3, 3)


class LocalBundleAdjustmentStep(VoStep):
    def __init__(self, K: np.ndarray):
        super().__init__("LocalBundleAdjustment")
        self.K = K
        self.config = LocalBundleAdjustmentConfig()

        # The sliding window buffer (deque efficiently handles pushes/pops)
        self._window: deque[WindowFrame] = deque(maxlen=self.config.window_size)

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
        if not self.config.enable_ba:
            return state.pose, state.P, state.X, None

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

        # C. Pre-compute Flat Arrays
        (obs_u, obs_v, pt_x_indices, pose_x_indices, is_optimized_mask) = (
            self._flatten_data_for_solver(valid_obs, pose_indices, landmark_indices)
        )

        # D. Run Optimization (Cached)
        anchor_frame = self._window[0]
        anchor_pose = self._optimized_poses.get(
            anchor_frame.frame_id, anchor_frame.pose
        )

        # Cache Container: [last_x_ref, last_residuals, last_jacobian]
        # We use a mutable list to let inner functions write to it
        cache = [None, None, None]

        def compute_common(x_in):
            # Check if we already computed for this X
            if cache[0] is not None and np.array_equal(x_in, cache[0]):
                return

            # Compute Both
            r, J = self._compute_residuals_and_jacobian(
                x_in,
                obs_u,
                obs_v,
                pt_x_indices,
                pose_x_indices,
                is_optimized_mask,
                anchor_pose,
            )

            # Update Cache
            cache[0] = x_in
            cache[1] = r
            cache[2] = J

        def fun_wrapper(x):
            compute_common(x)
            return cache[1]

        def jac_wrapper(x):
            compute_common(x)
            return cache[2]

        # Execute
        res = least_squares(
            fun=fun_wrapper,
            jac=jac_wrapper,  # <--- Pass the analytical Jacobian wrapper
            x0=x0,
            verbose=0,
            x_scale="jac",
            ftol=self.config.ftol,
            max_nfev=self.config.max_nfev,
            method="trf",
            loss=self.config.loss_function,
            f_scale=self.config.f_scale,
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

    def _compute_residuals_and_jacobian(
        self,
        x: np.ndarray,
        obs_u: np.ndarray,
        obs_v: np.ndarray,
        pt_x_indices: np.ndarray,
        pose_x_indices: np.ndarray,
        is_optimized_mask: np.ndarray,
        fixed_anchor_pose: np.ndarray,
    ):
        """
        Computes Residuals AND Jacobian simultaneously to reuse projection math.
        """
        # --- 1. SETUP & RECOVER STATE (Identical to Residuals) ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        N = len(obs_u)

        # Gather Points
        col_offsets = np.array([0, 1, 2])
        gather_indices = pt_x_indices[:, None] + col_offsets[None, :]
        points_3d = x[gather_indices]  # (N, 3)

        # Gather Poses
        R_anchor_WC = fixed_anchor_pose[:3, :3]
        t_anchor_WC = fixed_anchor_pose[:3, 3]
        R_anchor_CW = R_anchor_WC.T
        t_anchor_CW = -R_anchor_CW @ t_anchor_WC

        R_batch = np.tile(R_anchor_CW, (N, 1, 1))
        t_batch = np.tile(t_anchor_CW, (N, 1))

        if np.any(is_optimized_mask):
            opt_indices = pose_x_indices[is_optimized_mask]
            pose_offsets = np.arange(6)
            pose_gather = opt_indices[:, None] + pose_offsets[None, :]
            pose_params = x[pose_gather]

            r_vecs = pose_params[:, :3]
            t_vecs = pose_params[:, 3:]
            R_opt = _batched_rodrigues(r_vecs)

            R_batch[is_optimized_mask] = R_opt
            t_batch[is_optimized_mask] = t_vecs

        # --- 2. PROJECT (Shared Computation) ---
        # X_cam = R * X_w + t
        X_cam = np.einsum("nij,nj->ni", R_batch, points_3d) + t_batch  # (N, 3)

        z = X_cam[:, 2]
        valid_depth = z > 1e-4
        z_safe = np.where(valid_depth, z, 1.0)
        inv_z = 1.0 / z_safe
        inv_z2 = inv_z * inv_z

        # Residuals
        u_proj = fx * X_cam[:, 0] * inv_z + cx
        v_proj = fy * X_cam[:, 1] * inv_z + cy
        res_u = u_proj - obs_u
        res_v = v_proj - obs_v

        residuals = np.column_stack((res_u, res_v)).flatten()
        # Heavy penalty for invalid depth
        residuals[np.repeat(~valid_depth, 2)] = 100.0

        # --- 3. ANALYTICAL JACOBIAN ---

        # A. Projective Derivative (d_Projection / d_Xcam)
        # J_proj is (N, 2, 3)
        # [ fx/z   0   -fx*x/z^2 ]
        # [  0    fy/z -fy*y/z^2 ]

        J_proj = np.zeros((N, 2, 3))
        J_proj[:, 0, 0] = fx * inv_z
        J_proj[:, 0, 2] = -fx * X_cam[:, 0] * inv_z2
        J_proj[:, 1, 1] = fy * inv_z
        J_proj[:, 1, 2] = -fy * X_cam[:, 1] * inv_z2

        # B. Landmark Derivative (d_Residual / d_PointWorld)
        # Chain Rule: J_point = J_proj @ R_cw
        # (N, 2, 3) @ (N, 3, 3) -> (N, 2, 3)
        J_point = np.einsum("nij,njk->nik", J_proj, R_batch)

        # C. Pose Derivative (d_Residual / d_Pose)
        # Only calculated for optimized frames
        # We need (M, 2, 6) where M is count of is_optimized_mask

        if np.any(is_optimized_mask):
            # Subset relevant matrices
            J_proj_opt = J_proj[is_optimized_mask]  # (M, 2, 3)
            X_cam_opt = X_cam[is_optimized_mask]  # (M, 3)

            # 1. Translation Part (d_Xcam / d_t = Identity)
            # J_trans = J_proj * I = J_proj
            J_trans = J_proj_opt  # (M, 2, 3)

            # 2. Rotation Part (d_Xcam / d_w = -[X_cam]_x)
            # J_rot = J_proj @ -skew(X_cam)
            skew_X = _batched_skew_symmetric(X_cam_opt)
            J_rot = -np.einsum("nij,njk->nik", J_proj_opt, skew_X)  # (M, 2, 3)

            # Combine [J_rot, J_trans] -> (M, 2, 6)
            J_pose = np.concatenate([J_rot, J_trans], axis=2)

        # --- 4. CONSTRUCT SPARSE MATRIX ---
        # We build the COO matrix arrays

        # Row Indices: 0, 0, 0... 1, 1, 1...
        # Each observation i has 2 rows: 2*i, 2*i+1
        obs_indices = np.arange(N)
        rows_u = obs_indices * 2
        rows_v = obs_indices * 2 + 1

        # --- Fill Point Derivatives ---
        # J_point is (N, 2, 3). Flattening:
        # data order: [Pt0_u_x, Pt0_u_y, Pt0_u_z, Pt0_v_x, ... ]
        data_pt = J_point.reshape(N * 6)

        # Col Indices for Points
        # pt_x_indices is start index. We need [idx, idx+1, idx+2]
        # Repeat rows for 3 columns
        rows_pt = np.repeat(np.stack([rows_u, rows_v], axis=1), 3, axis=1).flatten()
        cols_pt = pt_x_indices[:, None] + np.arange(3)[None, :]  # (N, 3)
        cols_pt = np.repeat(cols_pt, 2, axis=0).flatten()  # Repeat for u and v

        # --- Fill Pose Derivatives ---
        # Only for optimized frames
        if np.any(is_optimized_mask):
            M = np.sum(is_optimized_mask)
            data_pose = J_pose.reshape(M * 12)  # 2 rows * 6 params

            # Rows for poses
            rows_u_opt = rows_u[is_optimized_mask]
            rows_v_opt = rows_v[is_optimized_mask]
            rows_pose = np.repeat(
                np.stack([rows_u_opt, rows_v_opt], axis=1), 6, axis=1
            ).flatten()

            # Cols for poses
            opt_pose_start = pose_x_indices[is_optimized_mask]  # (M,)
            cols_pose = opt_pose_start[:, None] + np.arange(6)[None, :]  # (M, 6)
            cols_pose = np.repeat(cols_pose, 2, axis=0).flatten()

            # Concatenate All
            all_data = np.concatenate([data_pt, data_pose])
            all_rows = np.concatenate([rows_pt, rows_pose])
            all_cols = np.concatenate([cols_pt, cols_pose])
        else:
            all_data = data_pt
            all_rows = rows_pt
            all_cols = cols_pt

        jacobian = coo_matrix((all_data, (all_rows, all_cols)), shape=(N * 2, len(x)))

        return residuals, jacobian

    def _flatten_data_for_solver(
        self,
        valid_observations: dict,
        pose_indices: dict[int, int],
        landmark_indices: dict[int, int],
    ):
        """
        Pre-computes flat arrays so the residual function performs NO Python loops.
        """
        obs_u_list = []
        obs_v_list = []
        pt_x_indices_list = []  # Indices in 'x' for 3D points
        pose_x_indices_list = []  # Indices in 'x' for Poses
        is_optimized_list = []  # Boolean: True if not Anchor

        # Iterate in the exact same order as Sparsity
        for lid, obs_list in valid_observations.items():
            idx_pt = landmark_indices[lid]

            for frame_idx, observed_uv in obs_list:
                obs_u_list.append(observed_uv[0])
                obs_v_list.append(observed_uv[1])
                pt_x_indices_list.append(idx_pt)

                if frame_idx == 0:
                    # Anchor Frame
                    is_optimized_list.append(False)
                    pose_x_indices_list.append(-1)  # Dummy value
                else:
                    # Optimized Frame
                    is_optimized_list.append(True)
                    pose_x_indices_list.append(pose_indices[frame_idx])

        # Convert to aligned NumPy arrays
        return (
            np.array(obs_u_list, dtype=np.float64),
            np.array(obs_v_list, dtype=np.float64),
            np.array(pt_x_indices_list, dtype=np.int32),
            np.array(pose_x_indices_list, dtype=np.int32),
            np.array(is_optimized_list, dtype=bool),
        )

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
        curr_map_3d = curr_map_3d = dict(zip(state.landmark_ids, state.X, strict=True))

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
        if len(self._window) == self.config.window_size:
            # The oldest frame (index 0) is about to fall off the edge
            removed_frame = self._window[0]

            # Remove from optimization cache to prevent memory leaks
            if removed_frame.frame_id in self._optimized_poses:
                del self._optimized_poses[removed_frame.frame_id]

        # Add to window (auto-pops oldest if full)
        self._window.append(frame_snapshot)
