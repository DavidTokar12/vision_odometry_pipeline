from __future__ import annotations

import os
import time

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from vision_odometry_pipeline.steps.create_undistorted_maps import (
    create_undistorted_maps,
)
from vision_odometry_pipeline.steps.intial_pose_finding import InitialPoseFindingStep
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
        D: np.ndarray,
        debug: bool = False,
        debug_output: str | Path | None = None,
    ):
        """
        The Orchestrator of the VO Pipeline.
        """
        self._debug = debug
        self._debug_out = debug_output

        # --- Internal State Management ---
        self._state = VoState()
        self._trajectory: list[np.ndarray] = []
        self._timings: dict[str, list[float]] = {}
        self._frame_idx = 0

        # --- Initialization Parameters ---
        self._calibration_matrix = K
        self._distortion_vec = D

        if self._debug and self._debug_out:
            os.makedirs(self._debug_out, exist_ok=True)

    def process_frame(self, image: np.ndarray) -> VoState:
        """
        Ingests a single image, pushes it through the pipeline steps.
        """

        if self._state.is_initialized == 0:
            h, w = image.shape[:2]

            # This function generates the look-up tables and the NEW optimal K
            map_x, map_y, roi, new_K = create_undistorted_maps(
                self._calibration_matrix, self._distortion_vec, (h, w)
            )

            # --- Instantiation of Pipeline Steps ---
            self.preproc = ImagePreprocessingStep(map_x, map_y, roi)
            self.tracker = KeypointTrackingStep()
            self.pose_est = PoseEstimationStep(K=new_K)
            self.triangulation = TriangulationStep(K=new_K)
            self.replenishment = ReplenishmentStep(max_features=200, min_dist=10)
            self.intial_pose_finding = InitialPoseFindingStep()

            # --- Preprocess Image ---
            self._state.image_buffer.update(image)
            gray_img, vis_pre = self.preproc.process(self._state, self._debug)
            self._state.image_buffer._buffer[-1] = gray_img

            # --- Find initial SIFT features and update state ---
            # TODO: check what is returned, might be related to candidates.
            first_P = self.intial_pose_finding.find_initial_features(self._state)
            self._state.is_initialized = replace(
                self._state, is_initialized=1, P=first_P
            )

        elif self._state.is_initialized == 1:
            # --- Preprocess Image ---
            self._state.image_buffer.update(image)
            gray_img, vis_pre = self.preproc.process(self._state, self._debug)
            self._state.image_buffer._buffer[-1] = gray_img

            # TODO: Complete this function, udpate state
            status = self.intial_pose_finding.process(self._state, self._debug)

            # if initial pose is found, do:
            if status == True:
                self._state = replace(self._state, is_initialized=2)

        else:
            t_total_start = time.perf_counter()

            # 0. Setup Debugging for this frame
            current_debug_dir = None
            if self._debug and self._debug_out:
                current_debug_dir = os.path.join(
                    self._debug_out, f"frame_{self._frame_idx:04d}"
                )
                os.makedirs(current_debug_dir, exist_ok=True)

            # ---------------------------------------------------------
            # STEP 1: Image Preprocessing
            # ---------------------------------------------------------
            # Special Case: We must load the raw image into buffer first
            self._state.image_buffer.update(image)

            t0 = time.perf_counter()
            gray_img, vis_pre = self.preproc.process(self._state, self._debug)
            self._record_timing("01_Preprocessing", t0)

            # Apply Result: Replace buffer content with Grayscale for tracking
            self._state.image_buffer._buffer[-1] = gray_img

            self._save_debug(current_debug_dir, "01_Preprocessing", vis_pre)

            # Early Exit: We need 2 frames to track
            if not self._state.image_buffer.is_ready:
                self._trajectory.append(self._state.pose.copy())
                self._frame_idx += 1
                return self._state

            # ---------------------------------------------------------
            # STEP 2: Keypoint Tracking (KLT)
            # ---------------------------------------------------------
            t0 = time.perf_counter()
            # Returns: (Filtered P, Filtered X, Filtered C, Filtered F, Filtered T, Vis)
            new_P, new_X, new_C, new_F, new_T, vis_track = self.tracker.process(
                self._state, self._debug
            )
            self._record_timing("02_Tracking", t0)

            self._state = replace(
                self._state, P=new_P, X=new_X, C=new_C, F=new_F, T_first=new_T
            )
            self._save_debug(current_debug_dir, "02_Tracking", vis_track)

            # ---------------------------------------------------------
            # STEP 3: Pose Estimation (P3P + RANSAC)
            # ---------------------------------------------------------
            t0 = time.perf_counter()
            # Returns: (New Pose, Inlier P, Inlier X, Vis)
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
            # Returns: (Full P, Full X, Remaining C, Remaining F, Remaining T, Vis)
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
            # Returns: (Full C, Full F, Full T, Vis)
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
