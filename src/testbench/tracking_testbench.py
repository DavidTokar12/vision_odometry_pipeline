import glob
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vision_odometry_pipeline.steps.key_point_tracker import KeypointTrackingStep
from vision_odometry_pipeline.steps.replenishment_step import ReplenishmentStep
from vision_odometry_pipeline.vo_state import VoState


class TrackingTestbench:
    def __init__(
        self,
        img_folder: str,
        output_folder: str,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        self.img_files = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        if not self.img_files:
            self.img_files = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))

        if not self.img_files:
            raise ValueError(f"No images found in {img_folder}")

        self.img_files = self.img_files[start_frame:end_frame]

        self.output_folder = output_folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        self.tracker = KeypointTrackingStep()
        self.replenisher = ReplenishmentStep()

        print(f"Input: {len(self.img_files)} images from '{img_folder}'")
        print(f"Output: Saving results to '{self.output_folder}'")

    def run(self):
        if len(self.img_files) < 2:
            print("Not enough images to test tracking (need at least 2).")
            return

        # Metrics Storage
        history_total = []
        history_tracked = []
        history_rate = []

        for i in range(len(self.img_files) - 1):
            path0 = self.img_files[i]
            path1 = self.img_files[i + 1]

            img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

            if img0 is None or img1 is None:
                continue

            # 1. Setup State
            state = VoState()
            state.image_buffer.update(img0)

            # 2. Run Replenishment on Frame 0
            # This fills state.C (and F, T) with fresh features
            # We clear P/C first just to be safe, though state is new
            object.__setattr__(state, "P", np.empty((0, 2)))
            object.__setattr__(state, "C", np.empty((0, 2)))

            # Run the actual replenishment step
            # It updates C, F, T internally or returns them.
            # The class method returns (C, F, T, Vis). We need to update state manually
            # because VoStep.process methods return data, they don't mutate state in place (usually).
            new_C, new_F, new_T, _ = self.replenisher.process(state, debug=False)

            # Update state with detected points
            object.__setattr__(state, "C", new_C)
            object.__setattr__(state, "F", new_F)
            object.__setattr__(state, "T_first", new_T)

            # Store initial points for visualization comparison later
            pts0 = new_C.copy()

            # 3. Update Buffer for Tracking
            # Buffer now needs [img0, img1]
            # state already has img0. Pushing img1 makes it:
            # prev=img0, curr=img1
            state.image_buffer.update(img1)

            # 4. Run Tracking
            # process() returns (New_P, New_X, New_ids, New_C, New_F, New_T, Vis)
            _, _, _, new_C_tracked, new_F_tracked, _, _ = self.tracker.process(
                state, debug=False
            )

            # 5. Analyze Results
            total_input = len(pts0)
            total_tracked = len(new_C_tracked)
            track_rate = (total_tracked / total_input) if total_input > 0 else 0.0

            history_total.append(total_input)
            history_tracked.append(total_tracked)
            history_rate.append(track_rate)

            # 6. Visualize
            vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

            # Draw Failures (Red)
            # All initial points
            for pt in pts0:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

            # Draw Successes (Green) - Overwrite Red
            if len(new_C_tracked) > 0:
                for start, end in zip(new_F_tracked, new_C_tracked, strict=True):
                    p_start = (int(start[0]), int(start[1]))
                    p_end = (int(end[0]), int(end[1]))
                    cv2.line(vis, p_start, p_end, (0, 255, 0), 1)
                    cv2.circle(vis, p_end, 2, (0, 255, 0), -1)

            # Text Info
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 60), (0, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            l1 = f"Frame {i} -> {i + 1}"
            l2 = f"Input: {total_input} | Tracked: {total_tracked}"
            l3 = f"Success Rate: {track_rate * 100:.1f}%"

            cv2.putText(vis, l1, (10, 18), font, 0.5, (255, 255, 255), 1)
            cv2.putText(vis, l2, (10, 36), font, 0.5, (255, 255, 255), 1)
            cv2.putText(
                vis,
                l3,
                (10, 54),
                font,
                0.5,
                (0, 255, 0) if track_rate > 0.9 else (0, 255, 255),
                1,
            )

            # Save
            out_name = os.path.join(self.output_folder, f"tracking_{i:04d}.png")
            cv2.imwrite(out_name, vis)
            print(f"Base frame: {i}, Rate: {track_rate * 100:.1f}%", end="\r")

        self.plot_results(history_total, history_tracked, history_rate)
        self.print_summary(history_total, history_tracked, history_rate)

    def plot_results(self, total, tracked, rate):
        """Plots tracking performance over frames."""
        frames = range(len(total))
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Feature Counts
        ax1.plot(frames, total, label="Input Features", color="gray", alpha=0.7)
        ax1.plot(frames, tracked, label="Successfully Tracked", color="green")
        ax1.set_title("Feature Tracking Counts")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Success Rate
        rate_percent = [r * 100 for r in rate]
        ax2.plot(frames, rate_percent, color="blue", label="Success Rate")
        ax2.axhline(y=95, color="green", linestyle="--", label="95% Threshold")
        ax2.axhline(y=80, color="red", linestyle="--", label="80% Threshold")
        ax2.set_title("Tracking Success Rate (%)")
        ax2.set_ylabel("Rate (%)")
        ax2.set_xlabel("Frame Index")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.output_folder, "A_tracking_metrics.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
        plt.close()

    def print_summary(self, h_tot, h_track, h_rate):
        if not h_tot:
            print("No data processed.")
            return

        print("\n" + "=" * 65)
        print(f"{'TRACKING PERFORMANCE SUMMARY':^65}")
        print("=" * 65)
        print(f"{'Metric':<15} | {'Average':<10} | {'Median':<10} | {'5% Worst':<10}")
        print("-" * 65)

        def get_stats(data, scale=1.0, invert_bad=False):
            p_worst = np.percentile(data, 95) if invert_bad else np.percentile(data, 5)
            return (np.mean(data) * scale, np.median(data) * scale, p_worst * scale)

        t_avg, t_med, t_low = get_stats(h_tot)
        print(f"{'Input Count':<15} | {t_avg:<10.1f} | {t_med:<10.1f} | {t_low:<10.1f}")

        k_avg, k_med, k_low = get_stats(h_track)
        print(
            f"{'Tracked Count':<15} | {k_avg:<10.1f} | {k_med:<10.1f} | {k_low:<10.1f}"
        )

        r_avg, r_med, r_low = get_stats(h_rate, 100.0)
        print(
            f"{'Success Rate(%)':<15} | {r_avg:<10.1f} | {r_med:<10.1f} | {r_low:<10.1f}"
        )

        print("=" * 65)
        print("\nInterpretation Guide:")
        print("-" * 65)
        print("1. Success Rate:")
        print("   - High (>95%): Excellent temporal consistency.")
        print("   - Low (<80%): Fast motion, blur, or dynamic objects.")
        print("2. 5% Worst:")
        print("   - Represents the lower bound performance (the 'difficult' frames).")
        print("=" * 65)


if __name__ == "__main__":
    INPUT_DIR = "data/kitti/05/image_0"
    OUTPUT_DIR = "src/testbench/tracking_results"

    # Run
    tb = TrackingTestbench(INPUT_DIR, OUTPUT_DIR, start_frame=0, end_frame=400)
    tb.run()
