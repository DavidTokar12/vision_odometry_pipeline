from __future__ import annotations

import os
import time

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

# --- Step Imports ---
from vision_odometry_pipeline.steps.initialization_step import InitializationStep
from vision_odometry_pipeline.steps.key_point_tracker import KeypointTrackingStep
from vision_odometry_pipeline.steps.pose_estimation import PoseEstimationStep
from vision_odometry_pipeline.steps.preprocessing import ImagePreprocessingStep
from vision_odometry_pipeline.steps.replenishment_step import ReplenishmentStep
from vision_odometry_pipeline.steps.triangulation import TriangulationStep
from vision_odometry_pipeline.vo_state import VoState


class VoRunner:
    def __init__(
        self,
        K: np.ndarray,
        undistortion_maps: tuple[np.ndarray, np.ndarray, tuple],
        debug: bool = False,
        debug_output: str | Path | None = None,
    ):
        """
        The Orchestrator of the VO Pipeline.

        Args:
            K: The NEW optimal camera matrix (undistorted/cropped).
            undistortion_maps: Tuple (map_x, map_y, roi) from create_undistorted_maps.
        """
        self._debug = debug
        self._debug_out = debug_output

        # --- Unpack Maps ---
        map_x, map_y, roi = undistortion_maps

        # --- Instantiation of Pipeline Steps ---
        self.preproc = ImagePreprocessingStep()
        self.initialization = InitializationStep()  # <--- NEW STEP
        self.tracker = KeypointTrackingStep()
        self.pose_est = PoseEstimationStep(K=K)
        self.triangulation = TriangulationStep(K=K)
        self.replenishment = ReplenishmentStep(max_features=200, min_dist=10)

        # --- Internal State Management ---
        # Initialize State with the Static Calibration Data
        self._state = VoState(
            K=K, map_x=map_x, map_y=map_y, roi=roi, calibration_matrix=K
        )

        self._trajectory: list[np.ndarray] = []
        self._timings: dict[str, list[float]] = {}
        self._frame_idx = 0

        if self._debug and self._debug_out:
            os.makedirs(self._debug_out, exist_ok=True)

    def process_frame(self, image: np.ndarray) -> VoState:
        """
        Ingests a single image, pushes it through the pipeline steps.
        """
        t_total_start = time.perf_counter()

        # 0. Setup Debugging for this frame
        current_debug_dir = None
        if self._debug and self._debug_out:
            current_debug_dir = os.path.join(
                self._debug_out, f"frame_{self._frame_idx:04d}"
            )
            os.makedirs(current_debug_dir, exist_ok=True)

        # ---------------------------------------------------------
        # STEP 1: Image Preprocessing (ALWAYS RUNS)
        # ---------------------------------------------------------
        # Load raw image into buffer
        self._state.image_buffer.update(image)

        t0 = time.perf_counter()
        gray_img, vis_pre = self.preproc.process(self._state, self._debug)
        self._record_timing("01_Preprocessing", t0)

        # Apply Result: Replace buffer content with Grayscale for processing
        # This ensures Init and Tracking use the clean, undistorted image
        self._state.image_buffer._buffer[-1] = gray_img

        self._save_debug(current_debug_dir, "01_Preprocessing", vis_pre)

        # ---------------------------------------------------------
        # PIPELINE BRANCHING
        # ---------------------------------------------------------

        # BRANCH A: INITIALIZATION
        if not self._state.is_initialized:
            t0 = time.perf_counter()

            # process() returns (current_image, debug_vis) and updates state IN-PLACE
            _, vis_init = self.initialization.process(self._state, self._debug)

            self._record_timing("00_Initialization", t0)
            self._save_debug(current_debug_dir, "00_Initialization", vis_init)

            if self._state.is_initialized:
                print(f"Frame {self._frame_idx}: System Initialized!")

                # ADAPTER: Map Initialization outputs to Tracking inputs
                # InitStep writes to: current_keypoints, landmarks
                # Tracker expects: P (2D points), X (3D landmarks)
                self._state = replace(
                    self._state,
                    P=self._state.current_keypoints,
                    X=self._state.landmarks,
                    # Reset candidates
                    C=np.empty((0, 2)),
                    F=np.empty((0, 2)),
                    T_first=np.empty((0, 3, 4)),
                )

                # Save initial pose to trajectory
                self._trajectory.append(self._state.pose.copy())
            else:
                # Still buffering or failed init
                print(f"Frame {self._frame_idx}: Buffering for initialization...")

            # Increment and exit, skipping tracking logic for this frame
            self._frame_idx += 1
            return self._state

        # BRANCH B: TRACKING LOOP (Runs only if initialized)

        # ---------------------------------------------------------
        # STEP 2: Keypoint Tracking (KLT)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        new_P, new_X, new_C, new_F, new_T, vis_track = self.tracker.process(
            self._state, self._debug
        )
        self._record_timing("02_Tracking", t0)

        self._state = replace(
            self._state, P=new_P, X=new_X, C=new_C, F=new_F, T_first=new_T
        )
        self._save_debug(current_debug_dir, "02_Tracking", vis_track)

        # ---------------------------------------------------------
        # STEP 3: Pose Estimation (PnP + RANSAC)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        new_pose, final_P, final_X, vis_pose = self.pose_est.process(
            self._state, self._debug
        )
        self._record_timing("03_PoseEstimation", t0)

        self._state = replace(self._state, pose=new_pose, P=final_P, X=final_X)
        self._save_debug(current_debug_dir, "03_PoseEstimation", vis_pose)

        # ---------------------------------------------------------
        # STEP 4: Triangulation (Mapping)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        full_P, full_X, rem_C, rem_F, rem_T, vis_map = self.triangulation.process(
            self._state, self._debug
        )
        self._record_timing("04_Triangulation", t0)

        self._state = replace(
            self._state, P=full_P, X=full_X, C=rem_C, F=rem_F, T_first=rem_T
        )
        self._save_debug(current_debug_dir, "04_Triangulation", vis_map)

        # ---------------------------------------------------------
        # STEP 5: Replenishment (New Candidates)
        # ---------------------------------------------------------
        t0 = time.perf_counter()
        full_C, full_F, full_T, vis_rep = self.replenishment.process(
            self._state, self._debug
        )
        self._record_timing("05_Replenishment", t0)

        self._state = replace(self._state, C=full_C, F=full_F, T_first=full_T)
        self._save_debug(current_debug_dir, "05_Replenishment", vis_rep)

        # ---------------------------------------------------------
        # Finish Frame
        # ---------------------------------------------------------
        self._record_timing("Total_Frame_Time", t_total_start)

        self._trajectory.append(self._state.pose.copy())
        self._frame_idx += 1

        return self._state

    def get_trajectory(self) -> np.ndarray:
        return np.array(self._trajectory)

    def _record_timing(self, step_name: str, t_start: float):
        duration_ms = (time.perf_counter() - t_start) * 1000
        if step_name not in self._timings:
            self._timings[step_name] = []
        self._timings[step_name].append(duration_ms)

    def _save_debug(self, folder: str | None, name: str, image: np.ndarray | None):
        if self._debug and folder and image is not None:
            fname = f"{name}.png"
            cv2.imwrite(os.path.join(folder, fname), image)
