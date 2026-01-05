from __future__ import annotations

import multiprocessing as mp
import time

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from vision_odometry_pipeline.vo_configs import InitializationConfig
from vision_odometry_pipeline.vo_configs import KeypointTrackingConfig
from vision_odometry_pipeline.vo_configs import LocalBundleAdjustmentConfig
from vision_odometry_pipeline.vo_configs import PoseEstimationConfig
from vision_odometry_pipeline.vo_configs import ReplenishmentConfig
from vision_odometry_pipeline.vo_configs import TriangulationConfig


@dataclass
class VoFrameResult:
    """Result from processing a single frame - only what's needed for recording."""

    frame_id: int
    P: np.ndarray  # [N, 2] tracked keypoints
    C: np.ndarray  # [M, 2] candidate keypoints
    X: np.ndarray  # [N, 3] 3D landmarks
    trajectory: np.ndarray  # [N, 4, 4] full trajectory
    processing_time_ms: float
    step_timings: dict[str, float]


class VoRunnerProcess:
    """
    Runs VoRunner in a separate process with zero-copy image transfer.
    """

    def __init__(
        self,
        config,  # Config object
        image_shape: tuple[int, int] | tuple[int, int, int],
        image_dtype: np.dtype = np.uint8,
        initial_frame: int = 0,
        debug: bool = False,
        num_buffers: int = 2,
    ):
        self.image_shape = image_shape
        self.image_dtype = np.dtype(image_dtype)
        self.num_buffers = num_buffers

        self.buffer_size = int(np.prod(image_shape) * self.image_dtype.itemsize)

        self._shm_buffers: list[shared_memory.SharedMemory] = []
        self._command_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._process: mp.Process | None = None

        self._next_buffer_idx = 0
        self._pending_count = 0

        self._config = {
            "K": config.K.copy(),
            "D": config.D.copy(),
            "image_shape": image_shape,
            "image_dtype": self.image_dtype.str,
            "initial_frame": initial_frame,
            "debug": debug,
            "debug_output": config.dataset.debug_output,
            "keypoint_tracking": config.keypoint_tracking.model_dump(),
            "initialization": config.initialization.model_dump(),
            "pose_estimation": config.pose_estimation.model_dump(),
            "replenishment": config.replenishment.model_dump(),
            "triangulation": config.triangulation.model_dump(),
            "local_bundle_adjustment": config.local_bundle_adjustment.model_dump(),
        }

        self._started = False

    def start(self) -> None:
        """Start the worker process."""
        if self._started:
            return

        for _ in range(self.num_buffers):
            shm = shared_memory.SharedMemory(create=True, size=self.buffer_size)
            self._shm_buffers.append(shm)

        self._command_queue = mp.Queue()
        self._result_queue = mp.Queue()

        self._config["shm_names"] = [shm.name for shm in self._shm_buffers]

        self._process = mp.Process(
            target=_worker_main,
            args=(self._config, self._command_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()
        self._started = True

    def submit_frame(self, frame_id: int, image: np.ndarray) -> None:
        """Submit a frame for processing (non-blocking if buffers available)."""
        if not self._started:
            raise RuntimeError("Process not started. Call start() first.")

        while self._pending_count >= self.num_buffers:
            self._result_queue.get()
            self._pending_count -= 1

        buffer_idx = self._next_buffer_idx
        shm = self._shm_buffers[buffer_idx]
        shm_array = np.ndarray(self.image_shape, dtype=self.image_dtype, buffer=shm.buf)
        np.copyto(shm_array, image)

        self._command_queue.put((buffer_idx, frame_id))

        self._next_buffer_idx = (self._next_buffer_idx + 1) % self.num_buffers
        self._pending_count += 1

    def get_result(self, timeout: float | None = None) -> VoFrameResult | None:
        """Get the next result (blocking)."""
        try:
            result = self._result_queue.get(timeout=timeout)
            self._pending_count -= 1
            return result
        except:
            return None

    def stop(self) -> None:
        """Stop the worker process."""
        if not self._started:
            return

        if self._command_queue:
            self._command_queue.put(None)

        if self._process:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()

        for shm in self._shm_buffers:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

        self._shm_buffers.clear()
        self._started = False

    def __enter__(self) -> VoRunnerProcess:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def _worker_main(
    config: dict,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Worker process entry point."""
    from multiprocessing import shared_memory

    from vision_odometry_pipeline.vo_runner import VoRunner

    shm_buffers = [
        shared_memory.SharedMemory(name=name) for name in config["shm_names"]
    ]

    runner = VoRunner(
        K=config["K"],
        D=config["D"],
        initial_frame=config["initial_frame"],
        debug=config["debug"],
        debug_output=config["debug_output"],
        keypoint_tracking_config=KeypointTrackingConfig.model_validate(
            config["keypoint_tracking"]
        ),
        initialization_config=InitializationConfig.model_validate(
            config["initialization"]
        ),
        pose_estimation_config=PoseEstimationConfig.model_validate(
            config["pose_estimation"]
        ),
        replenishment_config=ReplenishmentConfig.model_validate(
            config["replenishment"]
        ),
        triangulation_config=TriangulationConfig.model_validate(
            config["triangulation"]
        ),
        local_bundle_adjustment_config=LocalBundleAdjustmentConfig.model_validate(
            config["local_bundle_adjustment"]
        ),
    )

    image_shape = config["image_shape"]
    image_dtype = np.dtype(config["image_dtype"])

    try:
        while True:
            command = command_queue.get()

            if command is None:
                break

            buffer_idx, _ = command

            shm = shm_buffers[buffer_idx]
            image = np.ndarray(image_shape, dtype=image_dtype, buffer=shm.buf).copy()

            t_start = time.perf_counter()
            state = runner.process_frame(image)
            processing_time_ms = (time.perf_counter() - t_start) * 1000

            result = VoFrameResult(
                frame_id=state.frame_id,
                P=state.P.copy(),
                C=state.C.copy(),
                X=state.X.copy(),
                trajectory=runner.get_trajectory().copy(),
                processing_time_ms=processing_time_ms,
                step_timings=runner.get_last_timings(),
            )

            result_queue.put(result)

    finally:
        for shm in shm_buffers:
            shm.close()
