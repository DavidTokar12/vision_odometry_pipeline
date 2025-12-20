from __future__ import annotations

import logging
import os
import subprocess

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec


matplotlib.use("Agg")


from vision_odometry_pipeline.vo_state import VoState


logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)


class VoRecorder:
    def __init__(self, output_path: str, frame_rate: int = 20):
        self.output_path = output_path
        self.frame_rate = frame_rate

        # Internal History
        self.landmark_history: list[int] = []
        self.frame_count = 0
        self.video_writer = None

        # Setup the Figure similar to the screenshot
        # Layout: 2 Rows, 3 Cols
        # Top-Left (0, 0-2): Current Image
        # Right (0-2, 2): Local Trajectory (Tall)
        # Bottom-Left (1, 0): Landmark Count
        # Bottom-Middle (1, 1): Full Trajectory
        self.fig = plt.figure(figsize=(16, 9))  # , constrained_layout=True)
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.5, 1])

        self.ax_img = self.fig.add_subplot(gs[0, 0:2])
        self.ax_local = self.fig.add_subplot(gs[:, 2])
        self.ax_count = self.fig.add_subplot(gs[1, 0])
        self.ax_full = self.fig.add_subplot(gs[1, 1])

        self.fig.subplots_adjust(
            left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2
        )

        # Style tweaks
        for ax in [self.ax_local, self.ax_count, self.ax_full]:
            ax.grid(True, linestyle=":", alpha=0.6)

        # --- FIX START: Initialize plotting handles ---
        self.img_artist = None  # Will hold the image object
        self.scatter_tracked = None  # Will hold the green keypoints
        self.scatter_candidates = None  # Will hold the red keypoints

        # Set static properties once
        self.ax_img.axis("off")
        # --- FIX END ---

    def update(self, state: VoState, full_trajectory: np.ndarray):
        self.frame_count += 1
        num_landmarks = len(state.P)
        self.landmark_history.append(num_landmarks)

        # Convert BGR (OpenCV) to RGB (Matplotlib)
        curr_img_rgb = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # 1. Plot Current Image (Optimized to fix jitter)
        # ---------------------------------------------------------

        # A. Update the Title
        self.ax_img.set_title(f"Current Frame {state.frame_id}")

        # B. Handle Image Artist (Create once, then update)
        if self.img_artist is None:
            # First frame: Create the image object
            self.img_artist = self.ax_img.imshow(curr_img_rgb, cmap="gray")
        else:
            # Subsequent frames: Just update pixel data
            self.img_artist.set_data(curr_img_rgb)

        # C. Handle Scatter Points (Remove old, plot new)
        # We cannot use .set_data() easily for scatters if the number of points changes.
        # It is cleaner to remove the previous scatter artist and plot a new one.

        if self.scatter_tracked is not None:
            self.scatter_tracked.remove()
            self.scatter_tracked = None

        if self.scatter_candidates is not None:
            self.scatter_candidates.remove()
            self.scatter_candidates = None

        # Plot Tracked (Green)
        if len(state.P) > 0:
            self.scatter_tracked = self.ax_img.scatter(
                state.P[:, 0], state.P[:, 1], c="lime", s=3, marker="x", label="Tracked"
            )

        # Plot Candidates (Red)
        if len(state.C) > 0:
            self.scatter_candidates = self.ax_img.scatter(
                state.C[:, 0],
                state.C[:, 1],
                c="red",
                s=3,
                marker="x",
                label="Candidates",
            )

        # Manage Legend (Only create it once or if visibility changes)
        # Since we aren't clearing the axis, the legend persists.
        # If you need dynamic legends, you can regenerate it, but usually calling it once is fine.
        if (
            self.scatter_tracked or self.scatter_candidates
        ) and self.ax_img.get_legend() is None:
            self.ax_img.legend(loc="upper right", fontsize="small")

        # ---------------------------------------------------------
        # 2. Plot Local Trajectory (Right Column)
        # ... (Rest of your code remains the same) ...
        # 2. Plot Local Trajectory (Right Column)
        # ... (Rest of code remains unchanged) ...

        # 2. Plot Local Trajectory & Landmarks (Right Column)
        # Showing last 20 frames + Active Landmarks
        self.ax_local.clear()
        self.ax_local.set_title("Trajectory of last 20 frames")
        self.ax_local.set_xlabel("X [m]")
        self.ax_local.set_ylabel("Z [m]")
        self.ax_local.set_aspect("equal", adjustable="datalim")

        # Get last 20 poses
        lookback = 20
        if len(full_trajectory) > 0:
            # Trajectory is (N, 4, 4), we want X (0,3) and Z (2,3)
            local_traj = full_trajectory[-lookback:]
            tx = local_traj[:, 0, 3]
            tz = local_traj[:, 2, 3]
            self.ax_local.plot(tx, tz, "b-o", markersize=3, linewidth=1, label="Path")

            # Plot Active Landmarks (X vs Z)
            # state.X is (N, 3) -> col 0 is X, col 2 is Z
            if len(state.X) > 0:
                self.ax_local.scatter(
                    state.X[:, 0],
                    state.X[:, 2],
                    c="black",
                    s=1,
                    alpha=0.5,
                    label="Landmarks",
                )

            # In update(), for the local trajectory:
            # cx, cz = current_x, current_z # (Get these from your trajectory)
            radius_x = max(abs(tx[0] - tx[-1]) * 1.5, 5)
            radius_y = max(abs(tz[0] - tz[-1]) * 1.5, 5)

            self.ax_local.set_xlim(tx[-1] - radius_x, tx[-1] + radius_x)
            self.ax_local.set_ylim(tz[-1] - radius_y, tz[-1] + radius_y)

        # 3. Plot Landmark Count History (Bottom Left)
        self.ax_count.clear()
        self.ax_count.set_title("# tracked landmarks (last 20 frames)")
        self.ax_count.set_xlabel("Frame")
        self.ax_count.set_ylabel("Count")

        # Plot last 20 counts
        frames = np.arange(max(0, self.frame_count - lookback), self.frame_count)
        counts = self.landmark_history[-lookback:]
        self.ax_count.plot(frames, counts, "k-")
        self.ax_count.set_ylim(bottom=0)

        # 4. Plot Full Trajectory (Bottom Middle)
        self.ax_full.clear()
        self.ax_full.set_title("Full Trajectory")
        self.ax_full.set_xlabel("X [m]")
        self.ax_full.set_ylabel("Z [m]")
        self.ax_full.set_aspect("equal", adjustable="datalim")

        if len(full_trajectory) > 0:
            all_tx = full_trajectory[:, 0, 3]
            all_tz = full_trajectory[:, 2, 3]
            self.ax_full.plot(all_tx, all_tz, "b-", linewidth=1)

        # ---------------------------------------------------------
        # Render & Write to Video
        # ---------------------------------------------------------
        self.fig.canvas.draw()

        img_plot = np.asarray(self.fig.canvas.buffer_rgba())

        img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

        if self.video_writer is None:
            h, w = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.frame_rate, (w, h)
            )

        self.video_writer.write(img_bgr)

    def close(self):
        if self.video_writer:
            self.video_writer.release()
        plt.close(self.fig)

    def compress(self):
        """
        Compresses video using FFmpeg.

        Args:
            input_path (str): Path to the source video.
            output_path (str): Path where the compressed video will be saved.
        """

        video_path, ext = os.path.splitext(self.output_path)
        video_path_compressed = f"{video_path}_compressed{ext}"

        # Check if input exists
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Input file not found: {self.output_path}")

        # The "Original Command" structure
        # Using a list of strings is safer and cleaner than a single shell string
        command = [
            "ffmpeg",
            "-y",  # Overwrite output without asking
            "-i",
            self.output_path,  # Input file
            "-c:v",
            "libx264",  # Video codec H.264
            "-crf",
            "23",  # Constant Rate Factor (adjust for quality/size)
            "-preset",
            "medium",  # Encoding speed vs compression ratio
            "-c:a",
            "aac",  # Audio codec
            "-b:a",
            "128k",  # Audio bitrate
            video_path_compressed,  # Output file
        ]

        try:
            # Run the command
            # capture_output=True allows you to handle stdout/stderr if needed
            subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            print(f"Successfully compressed: {video_path_compressed}")
        except subprocess.CalledProcessError as e:
            print(f"Error during compression: {e.stderr.decode()}")
            raise
