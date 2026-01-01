from __future__ import annotations

import logging
import os
import subprocess

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec

from vision_odometry_pipeline.vo_runner_process import VoFrameResult


matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)


class VoRecorder:
    """Records VO pipeline output to video with trajectory visualization."""

    def __init__(
        self,
        output_path: str,
        frame_rate: int = 20,
        plot_ground_truth: bool = False,
        figsize: tuple[int, int] = (1920, 1080),
    ):
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.plot_ground_truth = plot_ground_truth

        self.ground_truth: np.ndarray | None = None
        self.first_frame: int | None = None
        self.landmark_history: list[int] = []
        self.frame_count = 0

        self._ffmpeg_process: subprocess.Popen | None = None
        self._fig_width = figsize[0]
        self._fig_height = figsize[1]

        self._local_xlim = (0.0, 1.0)
        self._local_ylim = (0.0, 1.0)
        self._full_xlim = (0.0, 1.0)
        self._full_ylim = (0.0, 1.0)
        self._smoothing = 0.3  # Lower = smoother transitions

        self._setup_figure()

    def _setup_figure(self) -> None:
        """Initialize matplotlib figure and axes."""
        dpi = 100
        self.fig = plt.figure(
            figsize=(self._fig_width / dpi, self._fig_height / dpi), dpi=dpi
        )

        # Layout: 2 Rows, 3 Cols
        # Top-Left (0, 0:2): Current Camera Image
        # Right (0:2, 2): Local Trajectory
        # Bottom-Left (1, 0): Landmark Count
        # Bottom-Middle (1, 1): Full Trajectory
        gs = GridSpec(
            2,
            3,
            figure=self.fig,
            height_ratios=[1.5, 1],
            width_ratios=[1, 1, 1.2],  # Make right column wider
        )

        self.ax_img = self.fig.add_subplot(gs[0, 0:2])
        self.ax_local = self.fig.add_subplot(gs[:, 2])  # Already spans full height
        self.ax_count = self.fig.add_subplot(gs[1, 0])
        self.ax_full = self.fig.add_subplot(gs[1, 1])

        self.fig.subplots_adjust(
            left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.25, hspace=0.2
        )

        for ax in [self.ax_local, self.ax_count, self.ax_full]:
            ax.grid(True, linestyle=":", alpha=0.6)

        self.ax_img.axis("off")

        self.img_artist = None
        self.scatter_tracked = None
        self.scatter_candidates = None

    def _start_ffmpeg(self) -> None:
        """Start ffmpeg process for video encoding."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self._fig_width}x{self._fig_height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.frame_rate),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            self.output_path,
        ]

        self._ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _smooth_limits(
        self,
        current: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        """Smoothly interpolate axis limits to reduce visual shaking."""
        new_min = current[0] + self._smoothing * (target[0] - current[0])
        new_max = current[1] + self._smoothing * (target[1] - current[1])
        return (new_min, new_max)

    def _compute_padded_limits(
        self,
        data_min: float,
        data_max: float,
        padding: float = 0.1,
        min_range: float = 1.0,
    ) -> tuple[float, float]:
        """Compute axis limits with padding and minimum range."""
        data_range = max(data_max - data_min, min_range)
        pad = data_range * padding
        center = (data_min + data_max) / 2
        half_range = data_range / 2 + pad
        return (center - half_range, center + half_range)

    def set_ground_truth(self, ground_truth: np.ndarray | None) -> None:
        """Set ground truth trajectory for plotting."""
        self.ground_truth = ground_truth

    def update(
        self,
        image: np.ndarray,
        result: VoFrameResult,
        full_trajectory: np.ndarray,
    ) -> None:
        """Update visualization with new frame data."""

        if self.first_frame is None:
            self.first_frame = result.frame_id
            self._start_ffmpeg()

        self.frame_count += 1
        self.landmark_history.append(len(result.P))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self._update_image(image_rgb, result)
        self._update_local_trajectory(result, full_trajectory)
        self._update_landmark_count()
        self._update_full_trajectory(result, full_trajectory)
        self._write_frame()

    def _update_image(self, image_rgb: np.ndarray, result: VoFrameResult) -> None:
        """Update camera image display with keypoints."""
        self.ax_img.set_title(f"Frame {result.frame_id}")

        if self.img_artist is None:
            self.img_artist = self.ax_img.imshow(image_rgb)
        else:
            self.img_artist.set_data(image_rgb)

        if self.scatter_tracked is not None:
            self.scatter_tracked.remove()
            self.scatter_tracked = None
        if self.scatter_candidates is not None:
            self.scatter_candidates.remove()
            self.scatter_candidates = None

        if len(result.P) > 0:
            self.scatter_tracked = self.ax_img.scatter(
                result.P[:, 0],
                result.P[:, 1],
                c="lime",
                s=3,
                marker="x",
                label="Tracked",
            )

        if len(result.C) > 0:
            self.scatter_candidates = self.ax_img.scatter(
                result.C[:, 0],
                result.C[:, 1],
                c="red",
                s=3,
                marker="x",
                label="Candidates",
            )

        if (
            self.scatter_tracked or self.scatter_candidates
        ) and not self.ax_img.get_legend():
            self.ax_img.legend(loc="upper right", fontsize="small")

    def _update_local_trajectory(
        self,
        result: VoFrameResult,
        full_trajectory: np.ndarray,
    ) -> None:
        """Update local trajectory plot (last 20 frames)."""
        self.ax_local.clear()
        self.ax_local.set_title("Local Trajectory (last 20 frames)")
        self.ax_local.set_xlabel("X [m]")
        self.ax_local.set_ylabel("Z [m]")
        self.ax_local.grid(True, linestyle=":", alpha=0.6)

        if len(full_trajectory) == 0:
            return

        lookback = 20
        local_traj = full_trajectory[-lookback:]
        tx = local_traj[:, 0, 3]
        tz = local_traj[:, 2, 3]

        self.ax_local.plot(tx, tz, "b-o", markersize=3, linewidth=1, label="Path")

        if len(result.X) > 0:
            self.ax_local.scatter(
                result.X[:, 0],
                result.X[:, 2],
                c="black",
                s=1,
                alpha=0.5,
                label="Landmarks",
            )

        if self.plot_ground_truth and self.ground_truth is not None:
            current_frame = result.frame_id
            start_frame = max(self.first_frame, current_frame - lookback + 1)
            end_frame = current_frame + 1

            if end_frame <= len(self.ground_truth):
                gt_local = self.ground_truth[start_frame:end_frame]
                self.ax_local.plot(
                    gt_local[:, 0],
                    gt_local[:, 1],
                    "r--",
                    linewidth=1.5,
                    label="Ground Truth",
                )

        spread_x = np.ptp(tx)
        spread_z = np.ptp(tz)
        radius = max(spread_x, spread_z, 2.0)  # Minimum radius of 2.0

        target_xlim = (tx[-1] - radius, tx[-1] + radius)
        target_ylim = (tz[-1] - radius, tz[-1] + radius)

        self._local_xlim = self._smooth_limits(self._local_xlim, target_xlim)
        self._local_ylim = self._smooth_limits(self._local_ylim, target_ylim)

        self.ax_local.set_xlim(self._local_xlim)
        self.ax_local.set_ylim(self._local_ylim)

    def _update_landmark_count(self) -> None:
        """Update landmark count history plot."""
        self.ax_count.clear()
        self.ax_count.set_title("Tracked Landmarks (last 20 frames)")
        self.ax_count.set_xlabel("Frame")
        self.ax_count.set_ylabel("Count")
        self.ax_count.grid(True, linestyle=":", alpha=0.6)

        lookback = 20
        counts = self.landmark_history[-lookback:]
        frames = np.arange(max(0, self.frame_count - len(counts)), self.frame_count)

        self.ax_count.plot(frames, counts, "k-", linewidth=1)
        self.ax_count.set_ylim(bottom=0, top=max(max(counts) * 1.2, 100))

    def _update_full_trajectory(
        self,
        result: VoFrameResult,
        full_trajectory: np.ndarray,
    ) -> None:
        """Update full trajectory plot."""
        self.ax_full.clear()
        self.ax_full.set_title("Full Trajectory")
        self.ax_full.set_xlabel("X [m]")
        self.ax_full.set_ylabel("Z [m]")
        self.ax_full.grid(True, linestyle=":", alpha=0.6)

        if len(full_trajectory) == 0:
            return

        all_tx = full_trajectory[:, 0, 3]
        all_tz = full_trajectory[:, 2, 3]

        self.ax_full.plot(all_tx, all_tz, "b-", linewidth=1, label="Estimated")

        if self.plot_ground_truth and self.ground_truth is not None:
            start_frame = self.first_frame
            end_frame = result.frame_id + 1

            if end_frame <= len(self.ground_truth):
                gt_full = self.ground_truth[start_frame:end_frame]
                self.ax_full.plot(
                    gt_full[:, 0],
                    gt_full[:, 1],
                    "r--",
                    linewidth=1.5,
                    label="Ground Truth",
                )

            self.ax_full.legend(loc="best", fontsize="small")

        target_xlim = self._compute_padded_limits(
            all_tx.min(), all_tx.max(), min_range=5.0
        )
        target_ylim = self._compute_padded_limits(
            all_tz.min(), all_tz.max(), min_range=5.0
        )

        self._full_xlim = self._smooth_limits(self._full_xlim, target_xlim)
        self._full_ylim = self._smooth_limits(self._full_ylim, target_ylim)

        self.ax_full.set_xlim(self._full_xlim)
        self.ax_full.set_ylim(self._full_ylim)
        self.ax_full.set_aspect("equal", adjustable="datalim")

    def _write_frame(self) -> None:
        """Render figure and write to video."""
        self.fig.canvas.draw()

        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img_rgb = img[:, :, :3]

        if self._ffmpeg_process and self._ffmpeg_process.stdin:
            self._ffmpeg_process.stdin.write(img_rgb.tobytes())

    def close(self) -> None:
        """Close video writer and cleanup."""
        if self._ffmpeg_process:
            if self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None

        plt.close(self.fig)
