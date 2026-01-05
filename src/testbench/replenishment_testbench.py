import glob
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vision_odometry_pipeline.steps.replenishment_step import ReplenishmentStep
from vision_odometry_pipeline.vo_state import VoState


class ReplenishmentTestbench:
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

        # Slice the file list based on requested range
        self.img_files = self.img_files[start_frame:end_frame]

        self.output_folder = output_folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        self.step = ReplenishmentStep()

        print(f"Input: {len(self.img_files)} images (Sliced range) from '{img_folder}'")
        print(f"Output: Saving results to '{self.output_folder}'")

    def get_feature_scores(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Computes the min eigenvalue (corner response) for each point.
        """
        if len(points) == 0:
            return np.array([])

        # Compute dense response map for the image
        response_map = cv2.cornerMinEigenVal(img, blockSize=3, ksize=3)

        # Extract values at point locations
        scores = []
        h, w = img.shape
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            # Bounds check
            if 0 <= x < w and 0 <= y < h:
                scores.append(response_map[y, x])
            else:
                scores.append(0.0)

        return np.array(scores)

    def calculate_distribution(
        self, img_shape, points: np.ndarray
    ) -> tuple[float, float]:
        h, w = img_shape
        rows = self.step.config.grid_rows
        cols = self.step.config.grid_cols

        dy = h // rows
        dx = w // cols

        grid_counts = np.zeros((rows, cols), dtype=int)

        for pt in points:
            c = min(int(pt[0] // dx), cols - 1)
            r = min(int(pt[1] // dy), rows - 1)
            grid_counts[r, c] += 1

        fill_rate = np.count_nonzero(grid_counts) / grid_counts.size
        std_dev = np.std(grid_counts)
        return fill_rate, std_dev

    def run(self):
        state = VoState()

        # Metrics Storage
        history_counts = []
        history_eigen = []
        history_fill = []
        history_std = []

        for f_idx, img_path in enumerate(self.img_files):
            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if curr_img is None:
                continue

            # Update State Buffer
            state.image_buffer.update(curr_img)

            # Reset points to force fresh replenishment calculation for this frame
            object.__setattr__(state, "P", np.empty((0, 2)))
            object.__setattr__(state, "C", np.empty((0, 2)))

            # Run Step (debug=False to disable internal drawing)
            full_C, _, _, _ = self.step.process(state, debug=False)

            # --- METRICS & SCORING ---
            scores = self.get_feature_scores(curr_img, full_C)
            n_features = len(full_C)

            # Calculate threshold for top 50%
            median_score = np.median(scores) if len(scores) > 0 else 0.0
            avg_score = np.mean(scores) if len(scores) > 0 else 0.0

            fill_rate, dist_std = self.calculate_distribution(curr_img.shape, full_C)

            # Store for summary
            history_counts.append(n_features)
            history_eigen.append(avg_score)
            history_fill.append(fill_rate)
            history_std.append(dist_std)

            # --- VISUALIZATION ---
            vis = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)

            # Draw Grid Lines
            h, w = curr_img.shape
            rows, cols = self.step.config.grid_rows, self.step.config.grid_cols
            dy, dx = h // rows, w // cols
            for i in range(1, cols):
                cv2.line(vis, (i * dx, 0), (i * dx, h), (50, 50, 50), 1)
            for i in range(1, rows):
                cv2.line(vis, (0, i * dy), (w, i * dy), (50, 50, 50), 1)

            # Draw Points Color-Coded
            # Green (Top 50%), Red (Bottom 50%)
            for pt, score in zip(full_C, scores, strict=True):
                color = (0, 255, 0) if score >= median_score else (0, 0, 255)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, color, -1)

            # Draw Info Panel
            cv2.rectangle(vis, (0, 0), (w, 80), (0, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_col = (255, 255, 255)

            l1 = f"Frame: {f_idx} | Count: {n_features}"
            l2 = f"Avg Eigen: {avg_score:.4f} | Median: {median_score:.4f}"
            l3 = f"Distribution: Fill {fill_rate * 100:.0f}% | StdDev {dist_std:.2f}"
            l4 = "Green: Top 50% | Red: Bottom 50%"

            cv2.putText(vis, l1, (10, 20), font, 0.5, txt_col, 1)
            cv2.putText(vis, l2, (10, 38), font, 0.5, txt_col, 1)
            cv2.putText(vis, l3, (10, 56), font, 0.5, txt_col, 1)
            cv2.putText(vis, l4, (10, 74), font, 0.5, (0, 255, 255), 1)

            # Save
            out_name = os.path.join(
                self.output_folder, f"replenishment_{f_idx:04d}.png"
            )
            cv2.imwrite(out_name, vis)

            # Console Log
            print(
                f"Frame {f_idx}: {n_features} features, median score: {median_score:.4f}, distribution: {fill_rate * 100:.0f}%",
                end="\r",
            )

        self.plot_results(history_counts, history_eigen, history_fill, history_std)
        self.print_summary(history_counts, history_eigen, history_fill, history_std)

    def plot_results(self, counts, eigen, fill, std):
        """Plots feature replenishment statistics."""
        frames = range(len(counts))
        _, axs = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Feature Count
        axs[0, 0].plot(frames, counts, color="blue")
        axs[0, 0].set_title("Feature Count")
        axs[0, 0].set_ylabel("Count")
        axs[0, 0].grid(True, alpha=0.5)

        # 2. Avg Eigen Value
        axs[0, 1].plot(frames, eigen, color="orange")
        axs[0, 1].set_title("Average Feature Response (MinEigenVal)")
        axs[0, 1].set_ylabel("Score")
        axs[0, 1].grid(True, alpha=0.5)

        # 3. Fill Rate
        fill_pct = [f * 100 for f in fill]
        axs[1, 0].plot(frames, fill_pct, color="green")
        axs[1, 0].set_title("Grid Fill Rate (%)")
        axs[1, 0].set_ylabel("Fill (%)")
        axs[1, 0].set_ylim(0, 100)
        axs[1, 0].grid(True, alpha=0.5)

        # 4. Std Dev
        axs[1, 1].plot(frames, std, color="red")
        axs[1, 1].set_title("Distribution StdDev (Lower is Better)")
        axs[1, 1].set_ylabel("StdDev")
        axs[1, 1].grid(True, alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(self.output_folder, "A_replenishment_metrics.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
        plt.close()

    def print_summary(self, history_counts, history_eigen, history_fill, history_std):
        print("\n" + "=" * 65)
        print(f"{'PERFORMANCE SUMMARY':^65}")
        print("=" * 65)
        print(f"{'Metric':<15} | {'Average':<10} | {'Median':<10} | {'5% Low':<10}")
        print("-" * 65)
        if history_counts:
            # Helper for clean formatting
            def get_stats(data, scale=1.0, invert_bad=False):
                p_worst = (
                    np.percentile(data, 95) if invert_bad else np.percentile(data, 5)
                )
                return (np.mean(data) * scale, np.median(data) * scale, p_worst * scale)

            c_avg, c_med, c_low = get_stats(history_counts)
            print(
                f"{'Feature Count':<15} | {c_avg:<10.1f} | {c_med:<10.1f} | {c_low:<10.1f}"
            )

            e_avg, e_med, e_low = get_stats(history_eigen)
            print(
                f"{'Goodness (Eig)':<15} | {e_avg:<10.5f} | {e_med:<10.5f} | {e_low:<10.5f}"
            )

            f_avg, f_med, f_low = get_stats(history_fill, 100)
            print(
                f"{'Fill Rate (%)':<15} | {f_avg:<10.1f} | {f_med:<10.1f} | {f_low:<10.1f}"
            )

            # StdDev: Higher is WORSE, so we want the 95th percentile as the "Worst" case
            s_avg, s_med, s_low = get_stats(history_std, invert_bad=True)
            print(
                f"{'Dist StdDev':<15} | {s_avg:<10.2f} | {s_med:<10.2f} | {s_low:<10.2f}"
            )
        else:
            print("No frames processed.")
        print("=" * 65 + "\n")

        # --- EXPLANATION ---
        rows = self.step.config.grid_rows
        cols = self.step.config.grid_cols

        print("\nInterpretation Guide:")
        print("-" * 65)
        print("1. '5% Worst' (Boundary):")
        print(
            "   - For Count, Goodness, Fill: The floor value for the worst 5% (Low = Bad)."
        )
        print(
            "   - For StdDev: The ceiling value for the worst 5% (High = Bad/Uneven)."
        )
        print("")
        print("2. Metrics:")
        print(
            "   - Goodness (Eigen): Average minEigenVal. >0.01 is usually a strong corner."
        )
        print(
            f"   - Fill Rate: % of {rows}x{cols} grid cells ({rows * cols} bins) with at least one point."
        )
        print(
            "   - Dist StdDev: Deviation of points per cell. Closer to 0.0 implies perfect uniformity."
        )
        print("=" * 65)


if __name__ == "__main__":
    # CONFIGURATION
    INPUT_DIR = "data/kitti/05/image_0"
    OUTPUT_DIR = "src/testbench/replenishment_results"

    # Dummy Generator for testing immediate functionality
    if not os.path.exists(INPUT_DIR):
        print(f"Generating dummy data in {INPUT_DIR}...")
        os.makedirs(INPUT_DIR, exist_ok=True)
        for i in range(5):
            # Create a noisy image
            dummy = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
            # Add strong corners (white squares)
            for _ in range(200):
                rx, ry = np.random.randint(10, 630), np.random.randint(10, 470)
                dummy[ry - 2 : ry + 2, rx - 2 : rx + 2] = 255
            cv2.imwrite(os.path.join(INPUT_DIR, f"{i:04d}.png"), dummy)

    tb = ReplenishmentTestbench(INPUT_DIR, OUTPUT_DIR, start_frame=0, end_frame=1000)
    tb.run()
    print("Done.")
