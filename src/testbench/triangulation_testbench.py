import glob
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vision_odometry_pipeline.steps.key_point_tracker import KeypointTrackingStep
from vision_odometry_pipeline.steps.replenishment_step import ReplenishmentStep
from vision_odometry_pipeline.steps.triangulation import TriangulationStep
from vision_odometry_pipeline.vo_state import VoState


class TriangulationTestbench:
    def __init__(
        self,
        img_folder: str,
        pose_file: str,
        output_folder: str,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        # 1. Load Images
        self.img_files = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        if not self.img_files:
            self.img_files = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))

        # 2. Load GT Poses
        self.gt_poses = self.load_kitti_poses(pose_file)

        # Slice Data
        self.img_files = self.img_files[start_frame:end_frame]
        self.gt_poses = self.gt_poses[start_frame:end_frame]

        if len(self.img_files) != len(self.gt_poses):
            print(
                f"Warning: Image count ({len(self.img_files)}) != GT Pose count ({len(self.gt_poses)})"
            )
            # Truncate to min
            min_len = min(len(self.img_files), len(self.gt_poses))
            self.img_files = self.img_files[:min_len]
            self.gt_poses = self.gt_poses[:min_len]

        # 3. Setup Output
        self.output_folder = output_folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

        # 4. Calibration
        self.K = np.array(
            [
                [7.18856e02, 0, 6.071928e02],
                [0, 7.18856e02, 1.852157e02],
                [0, 0, 1],
            ]
        )

        # 5. Initialize Steps
        self.tracker = KeypointTrackingStep()
        self.replenisher = ReplenishmentStep()

        # We pass K to TriangulationStep as required
        self.triangulator = TriangulationStep(self.K)

        print(f"Input: {len(self.img_files)} frames.")
        print(f"Output: {self.output_folder}")

    def load_kitti_poses(self, path):
        poses = []
        with open(path) as f:
            for line in f:
                val = [float(v) for v in line.split()]
                P = np.array(val).reshape(3, 4)
                T = np.eye(4)
                T[:3, :] = P
                poses.append(T)
        return poses

    def _inv_pose(self, T):
        """Inverts a 4x4 pose matrix (T_wc -> T_cw or vice versa)."""
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    def calculate_reprojection_error(self, P, X, pose, K):
        if len(P) == 0:
            return 0.0

        # Pose is T_cw (World -> Camera)
        R_cw = pose[:3, :3]
        t_cw = pose[:3, 3]

        # Transform World point X to Camera Frame: X_cam = R * X + t
        X_cam = (R_cw @ X.T).T + t_cw

        # Project: u = fx * X/Z + cx
        z = X_cam[:, 2] + 1e-9
        u = (X_cam[:, 0] * K[0, 0] / z) + K[0, 2]
        v = (X_cam[:, 1] * K[1, 1] / z) + K[1, 2]
        P_proj = np.stack([u, v], axis=1)

        return np.mean(np.linalg.norm(P - P_proj, axis=1))

    def run(self):
        if len(self.img_files) < 2:
            print("Need at least 2 frames.")
            return

        state = VoState()

        # Metrics History
        hist_new_lm = []  # Count of new landmarks per frame
        hist_repr_err = []  # Reprojection error of ALL landmarks
        hist_avg_depth = []  # Average depth of ALL landmarks

        # --- FRAME 0: Initialization ---
        # Just detection, no triangulation possible yet
        img0 = cv2.imread(self.img_files[0], cv2.IMREAD_GRAYSCALE)
        state.image_buffer.update(img0)

        # Set GT Pose for Frame 0
        object.__setattr__(state, "pose", self.gt_poses[0])

        # Replenish to get initial Candidates
        c0, f0, _, _ = self.replenisher.process(state, debug=False)
        object.__setattr__(state, "C", c0)
        object.__setattr__(state, "F", f0)

        # Set T_first for these candidates to the GT Pose of Frame 0
        # (This is crucial for triangulation later)
        pose0_flat = self.gt_poses[0][:3, :].flatten()
        t_first = np.tile(pose0_flat, (len(c0), 1))
        object.__setattr__(state, "T_first", t_first)

        # --- MAIN LOOP ---
        for i in range(1, len(self.img_files)):
            img = cv2.imread(self.img_files[i], cv2.IMREAD_GRAYSCALE)
            if img is None:
                break

            # 1. Update Buffer & Pose
            state.image_buffer.update(img)
            gt_pose_curr = self.gt_poses[i]
            object.__setattr__(state, "pose", gt_pose_curr)

            # 2. Tracking
            # nP = Points successfully tracked from previous frame
            nP, nX, nIds, nC, nF, nT, _ = self.tracker.process(state, debug=False)

            # Update state with TRACKED points
            object.__setattr__(state, "P", nP)
            object.__setattr__(state, "X", nX)
            object.__setattr__(state, "landmark_ids", nIds)
            object.__setattr__(state, "C", nC)
            object.__setattr__(state, "F", nF)
            object.__setattr__(state, "T_first", nT)

            # BUG FIX: Measure 'Pre-Triangulation' count here, AFTER tracking
            count_pre_tri = len(nP)

            # 3. Triangulation
            full_P, full_X, full_ids, rem_C, rem_F, rem_T, _ = (
                self.triangulator.process(state, debug=False)
            )

            # Update state with NEW + OLD points
            object.__setattr__(state, "P", full_P)
            object.__setattr__(state, "X", full_X)
            object.__setattr__(state, "landmark_ids", full_ids)
            object.__setattr__(state, "C", rem_C)
            object.__setattr__(state, "F", rem_F)
            object.__setattr__(state, "T_first", rem_T)

            # 4. Replenishment
            full_C, full_F, full_T, _ = self.replenisher.process(state, debug=False)
            object.__setattr__(state, "C", full_C)
            object.__setattr__(state, "F", full_F)
            object.__setattr__(state, "T_first", full_T)

            # --- METRICS ---
            # Correct Metric: (Tracked + New) - (Tracked) = New
            new_pts = len(full_P) - count_pre_tri

            repr_err = self.calculate_reprojection_error(
                full_P, full_X, gt_pose_curr, self.K
            )

            # Depth Stats (Using T_cw directly)
            R_cw = gt_pose_curr[:3, :3]
            t_cw = gt_pose_curr[:3, 3]
            X_cam = (R_cw @ full_X.T).T + t_cw
            avg_depth = np.mean(X_cam[:, 2]) if len(X_cam) > 0 else 0.0

            hist_new_lm.append(new_pts)
            hist_repr_err.append(repr_err)
            hist_avg_depth.append(avg_depth)

            # --- VISUALIZATION ---
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for pt in full_P:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            for pt in full_C:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), -1)

            cv2.rectangle(vis, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(
                vis,
                f"Frame {i} | Map: {len(full_X)} | New: {new_pts}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                vis,
                f"Reproj Err: {repr_err:.3f} px",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if repr_err < 1.0 else (0, 0, 255),
                1,
            )

            out_name = os.path.join(self.output_folder, f"tri_{i:04d}.png")
            cv2.imwrite(out_name, vis)

            print(
                f"Frame {i} | Map: {len(full_X)} | Err: {repr_err:.3f} px | Depth: {avg_depth:.1f} m",
                end="\r",
            )

        self.plot_results(hist_new_lm, hist_repr_err, hist_avg_depth)
        self.print_summary(hist_new_lm, hist_repr_err, hist_avg_depth)

    def plot_results(self, new_lm, repr_err, depths):
        """Plots triangulation metrics over frames."""
        frames = range(len(new_lm))
        _, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # New Landmarks
        axs[0].plot(frames, new_lm, color="blue", label="New Landmarks")
        axs[0].set_title("New Landmarks per Frame")
        axs[0].set_ylabel("Count")
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # Reprojection Error
        axs[1].plot(frames, repr_err, color="red", label="Reprojection Error")
        axs[1].axhline(y=1.0, color="green", linestyle="--", label="1.0 px Threshold")
        axs[1].set_title("Reprojection Error (px)")
        axs[1].set_ylabel("Error (px)")
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.6)

        # Average Depth
        axs[2].plot(frames, depths, color="purple", label="Avg Depth")
        axs[2].set_title("Average Landmark Depth (m)")
        axs[2].set_ylabel("Depth (m)")
        axs[2].set_xlabel("Frame Index")
        axs[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.output_folder, "A_triangulation_metrics.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
        plt.close()

    def print_summary(self, new_lm, repr_err, depths):
        print("\n" + "=" * 65)
        print(f"{'TRIANGULATION PERFORMANCE (with GT Poses)':^65}")
        print("=" * 65)
        print(f"{'Metric':<20} | {'Mean':<10} | {'Median':<10} | {'5% Worst':<10}")
        print("-" * 65)

        def get_stats(data, invert_bad=False):
            if not data:
                return 0.0, 0.0, 0.0
            p_worst = np.percentile(data, 95) if invert_bad else np.percentile(data, 5)
            return np.mean(data), np.median(data), p_worst

        # For New Landmarks, Low is Bad
        n_avg, n_med, n_low = get_stats(new_lm, invert_bad=False)
        print(
            f"{'New Landmarks/Fr':<20} | {n_avg:<10.1f} | {n_med:<10.1f} | {n_low:<10.1f}"
        )

        # For Reproj Error, High is Bad
        e_avg, e_med, e_high = get_stats(repr_err, invert_bad=True)
        print(
            f"{'Reproj Error (px)':<20} | {e_avg:<10.3f} | {e_med:<10.3f} | {e_high:<10.3f}"
        )

        # For Depth, extremes are suspicious, but generally just informational
        d_avg, d_med, d_high = get_stats(depths, invert_bad=True)
        print(
            f"{'Avg Depth (m)':<20} | {d_avg:<10.1f} | {d_med:<10.1f} | {d_high:<10.1f}"
        )

        print("=" * 65)
        print("Interpretation:")
        print("1. Reproj Error: < 1.0px is ideal. High error means triangulation is")
        print(
            "   generating points that don't match observations (bad calibration or outliers)."
        )
        print(
            "2. New Landmarks: If this is 0 often, check 'min_angle_deg' or 'min_pixel_dist'."
        )
        print("3. Avg Depth: If huge (>100m) or negative, check 'max_depth' filtering.")
        print("=" * 65)


if __name__ == "__main__":
    IMG_DIR = "data/kitti/05/image_0"
    POSE_FILE = "data/kitti/poses/05.txt"
    OUT_DIR = "src/testbench/triangulation_results"

    tb = TriangulationTestbench(
        IMG_DIR, POSE_FILE, OUT_DIR, start_frame=0, end_frame=400
    )
    tb.run()
