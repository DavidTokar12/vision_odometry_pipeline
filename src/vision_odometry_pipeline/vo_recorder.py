from __future__ import annotations

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec


matplotlib.use("Agg")

from vision_odometry_pipeline.vo_state import VoState


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
        self.fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.5, 1])

        self.ax_img = self.fig.add_subplot(gs[0, 0:2])
        self.ax_local = self.fig.add_subplot(gs[:, 2])
        self.ax_count = self.fig.add_subplot(gs[1, 0])
        self.ax_full = self.fig.add_subplot(gs[1, 1])

        # Style tweaks
        for ax in [self.ax_local, self.ax_count, self.ax_full]:
            ax.grid(True, linestyle=":", alpha=0.6)

    def update(self, state: VoState, full_trajectory: np.ndarray):
        """
        Updates the dashboard with the current state and writes a frame to video.
        """
        self.frame_count += 1
        num_landmarks = len(state.P)
        self.landmark_history.append(num_landmarks)

        # 1. Plot Current Image (Top Left)
        self.ax_img.clear()
        self.ax_img.set_title(f"Current Frame {state.frame_id}")

        # Convert BGR (OpenCV) to RGB (Matplotlib)
        curr_img_rgb = cv2.cvtColor(state.image_buffer.curr, cv2.COLOR_BGR2RGB)
        self.ax_img.imshow(curr_img_rgb, cmap="gray")

        # Overlay Keypoints (P = Green, C = Red)
        if len(state.P) > 0:
            self.ax_img.scatter(
                state.P[:, 0], state.P[:, 1], c="lime", s=3, marker="x", label="Tracked"
            )
        if len(state.C) > 0:
            self.ax_img.scatter(
                state.C[:, 0],
                state.C[:, 1],
                c="red",
                s=3,
                marker="x",
                label="Candidates",
            )

        if len(state.P) > 0 or len(state.C) > 0:
            self.ax_img.legend(loc="upper right", fontsize="small")
        
        self.ax_img.axis("off")

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
