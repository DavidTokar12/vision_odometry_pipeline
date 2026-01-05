import glob
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vision_odometry_pipeline.steps.key_point_tracker import KeypointTrackingStep
from vision_odometry_pipeline.steps.pose_estimation import PoseEstimationStep
from vision_odometry_pipeline.steps.replenishment_step import ReplenishmentStep
from vision_odometry_pipeline.steps.triangulation import TriangulationStep
from vision_odometry_pipeline.vo_state import VoState


class PoseTestbench:
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

        self.img_files = self.img_files[start_frame:end_frame]

        # 2. Load GT Poses
        self.gt_poses = self.load_kitti_poses(pose_file)
        # Slice GT poses to match image slice (assuming 1-to-1 mapping)
        self.gt_poses = self.gt_poses[start_frame:end_frame]

        if len(self.img_files) != len(self.gt_poses):
            print(
                f"Warning: Image count ({len(self.img_files)}) != GT Pose count ({len(self.gt_poses)})"
            )

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
        self.pose_est = PoseEstimationStep(self.K)
        self.triangulator = TriangulationStep(self.K)
        self.replenisher = ReplenishmentStep()

    def load_kitti_poses(self, path):
        """Loads KITTI format poses (12 floats per line: r11 r12 r13 tx ...)."""
        poses = []
        with open(path) as f:
            for line in f:
                val = [float(v) for v in line.split()]
                # Reshape to 3x4
                P = np.array(val).reshape(3, 4)
                # Convert to 4x4
                T = np.eye(4)
                T[:3, :] = P
                poses.append(T)
        return poses

    def get_rotation_error(self, R_est, R_gt):
        """Computes rotation error in degrees."""
        # R_diff = R_est * R_gt^T
        R_diff = R_est @ R_gt.T
        # Angle = arccos((trace(R) - 1) / 2)
        trace = np.trace(R_diff)
        trace = np.clip(trace, -1.0, 3.0)
        rad = np.arccos((trace - 1) / 2)
        return np.degrees(rad)

    def bootstrap(self, state, img0, img1, T_gt_0, T_gt_1):
        """
        Force-initializes the map using GT poses for the first two frames.
        This ensures we start with the CORRECT SCALE.
        """
        # 1. Detect on Frame 0
        state.image_buffer.update(img0)
        object.__setattr__(state, "P", np.empty((0, 2)))
        object.__setattr__(state, "C", np.empty((0, 2)))
        c0, f0, _, _ = self.replenisher.process(state, debug=False)
        object.__setattr__(state, "C", c0)
        object.__setattr__(state, "F", f0)
        # IMPORTANT: Set T_first to the ACTUAL GT pose of frame 0
        # Flatten top 3x4 of GT pose
        t_first_gt = np.tile(T_gt_0[:3, :].flatten(), (len(c0), 1))
        object.__setattr__(state, "T_first", t_first_gt)

        # 2. Track to Frame 1
        state.image_buffer.update(img1)
        # Tracker returns (P, X, ids, C, F, T, vis)
        # Note: X is empty here because we haven't triangulated yet
        _, _, _, nC, nF, nT, _ = self.tracker.process(state, debug=False)

        # Update State
        object.__setattr__(state, "C", nC)
        object.__setattr__(state, "F", nF)
        object.__setattr__(state, "T_first", nT)
        object.__setattr__(state, "pose", T_gt_1)  # Force GT Pose for Frame 1

        # 3. Triangulate
        # This creates the initial 'state.P' and 'state.X' from Candidates
        # Triangulator returns (P, X, ids, rem_C, rem_F, rem_T, vis)
        fP, fX, fIds, rC, rF, rT, _ = self.triangulator.process(state, debug=False)

        # Update State with new 3D points
        object.__setattr__(state, "P", fP)
        object.__setattr__(state, "X", fX)
        object.__setattr__(state, "landmark_ids", fIds)
        object.__setattr__(state, "C", rC)
        object.__setattr__(state, "F", rF)
        object.__setattr__(state, "T_first", rT)

        print(f"Bootstrap complete. Initialized {len(fX)} landmarks.")

    def run(self):
        if len(self.img_files) < 3:
            print("Need at least 3 frames.")
            return

        # Metrics
        history_r_err = []
        history_t_err_scale_corrected = []
        traj_est = []
        traj_gt = []
        history_scale_ratio = []

        state = VoState()

        # --- BOOTSTRAP (Frames 0 and 1) ---
        img0 = cv2.imread(self.img_files[0], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(self.img_files[1], cv2.IMREAD_GRAYSCALE)

        self.bootstrap(state, img0, img1, self.gt_poses[0], self.gt_poses[1])

        # Add initial poses to trajectory
        traj_est.extend([self.gt_poses[0], self.gt_poses[1]])
        traj_gt.extend([self.gt_poses[0], self.gt_poses[1]])

        # Initialize Previous Poses for Relative Error Calculation
        prev_gt_pose = self.gt_poses[1]
        prev_est_pose = state.pose.copy()

        # Frame ID tracker
        # Start at 2 because 0 and 1 were bootstrapped
        for i in range(2, len(self.img_files)):
            img_path = self.img_files[i]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                break

            # 1. Update Buffer
            state.image_buffer.update(img)

            # 2. Tracking
            # Matches 2D points in prev img to curr img
            # existing P -> P' (Corresponds to existing X)
            p_new, x_new, ids_new, c_new, f_new, t_new, _ = self.tracker.process(
                state, debug=False
            )

            # Update state temporarily for Pose Estimation
            object.__setattr__(state, "P", p_new)
            object.__setattr__(state, "X", x_new)
            object.__setattr__(state, "landmark_ids", ids_new)
            object.__setattr__(state, "C", c_new)
            object.__setattr__(state, "F", f_new)
            object.__setattr__(state, "T_first", t_new)

            # 3. POSE ESTIMATION (Target Step)
            # Uses P and X to find new pose
            pose_new, inlier_P, inlier_X, inlier_ids, _ = self.pose_est.process(
                state, debug=False
            )

            # Update State with new Pose and Inliers
            object.__setattr__(state, "pose", pose_new)
            object.__setattr__(state, "P", inlier_P)
            object.__setattr__(state, "X", inlier_X)
            object.__setattr__(state, "landmark_ids", inlier_ids)

            # 4. Triangulation (To maintain map)
            tP, tX, tIds, remC, remF, remT, _ = self.triangulator.process(
                state, debug=False
            )
            object.__setattr__(state, "P", tP)
            object.__setattr__(state, "X", tX)
            object.__setattr__(state, "landmark_ids", tIds)
            object.__setattr__(state, "C", remC)
            object.__setattr__(state, "F", remF)
            object.__setattr__(state, "T_first", remT)

            # 5. Replenishment (To maintain tracks)
            fullC, fullF, fullT, _ = self.replenisher.process(state, debug=False)
            object.__setattr__(state, "C", fullC)
            object.__setattr__(state, "F", fullF)
            object.__setattr__(state, "T_first", fullT)

            # --- METRICS & VISUALIZATION ---
            gt_pose = self.gt_poses[i]
            est_pose = state.pose

            # Store for final summary
            traj_est.append(est_pose)
            traj_gt.append(gt_pose)

            # --- RELATIVE ERROR CALCULATION ---
            # Compute relative motion (T_prev -> T_curr) in Camera Frame
            # T_rel = inv(T_prev) @ T_curr
            rel_gt = np.linalg.inv(prev_gt_pose) @ gt_pose
            rel_est = np.linalg.inv(prev_est_pose) @ est_pose

            # Relative Rotation Error
            r_err = self.get_rotation_error(rel_est[:3, :3], rel_gt[:3, :3])
            history_r_err.append(r_err)

            # Relative Translation Error (Scale Corrected)
            # We measure PnP accuracy (direction/shape) by normalizing the step length
            t_rel_est = rel_est[:3, 3]
            t_rel_gt = rel_gt[:3, 3]

            mag_est = np.linalg.norm(t_rel_est)
            mag_gt = np.linalg.norm(t_rel_gt)

            # Avoid division by zero
            step_scale = mag_gt / mag_est if mag_est > 1e-7 else 1.0

            # Compare the ESTIMATED step (scaled to match GT length) vs GT step
            # This isolates direction error from accumulated map scale error
            t_err_rel = np.linalg.norm(t_rel_est * step_scale - t_rel_gt)

            history_t_err_scale_corrected.append(t_err_rel)
            history_scale_ratio.append(step_scale)

            # Update Previous Poses
            prev_gt_pose = gt_pose
            prev_est_pose = est_pose.copy()

            # Output to Console
            print(
                f"Frame {i} | Inliers: {len(inlier_P)} | R_Err(Rel): {r_err:.3f}° | T_Err(Rel): {t_err_rel:.3f}m | Step Scale: {step_scale:.3f}",
                end="\r",
            )

            # Draw
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Draw Inliers
            for pt in inlier_P:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

            # Text
            cv2.rectangle(vis, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(
                vis,
                f"Frame {i} | PnP Inliers: {len(inlier_P)}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                vis,
                f"Rot Err: {r_err:.2f} deg",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                vis,
                f"Scale Drift: {step_scale:.3f}x",
                (200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            cv2.imwrite(os.path.join(self.output_folder, f"pose_{i:04d}.png"), vis)

        self.plot_results(
            history_r_err, history_t_err_scale_corrected, history_scale_ratio
        )
        self.print_summary(
            history_r_err, history_t_err_scale_corrected, history_scale_ratio
        )

    def plot_results(self, r_err, t_err, scale_ratio):
        """Plots pose estimation errors and scale drift."""
        frames = range(len(r_err))
        _, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Rotation Error
        axs[0].plot(frames, r_err, color="orange", label="Rotation Error")
        axs[0].set_title("Relative Rotation Error (deg)")
        axs[0].set_ylabel("Error (deg)")
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # Translation Error (Scale Corrected)
        axs[1].plot(frames, t_err, color="blue", label="Translation Error")
        axs[1].set_title("Relative Translation Error (Scale Corrected) (m)")
        axs[1].set_ylabel("Error (m)")
        axs[1].grid(True, linestyle="--", alpha=0.6)

        # Scale Ratio
        axs[2].plot(frames, scale_ratio, color="green", label="Scale Ratio (GT/Est)")
        axs[2].axhline(y=1.0, color="black", linestyle="--", label="Ideal (1.0)")
        axs[2].set_title("Scale Ratio (GT Step / Est Step)")
        axs[2].set_ylabel("Ratio")
        axs[2].set_xlabel("Frame Index")
        axs[2].legend()
        axs[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.output_folder, "A_pose_metrics.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
        plt.close()

    def print_summary(self, r_errs, t_errs, scales):
        print("\n" + "=" * 65)
        print(f"{'POSE ESTIMATION SUMMARY':^65}")
        print("=" * 65)
        print(f"{'Metric':<20} | {'Mean':<10} | {'Median':<10} | {'Max':<10}")
        print("-" * 65)

        def stats(data):
            return np.mean(data), np.median(data), np.max(data)

        if r_errs:
            rm, rmed, rmax = stats(r_errs)
            print(
                f"{'Rot Error (deg)':<20} | {rm:<10.3f} | {rmed:<10.3f} | {rmax:<10.3f}"
            )

            tm, tmed, tmax = stats(t_errs)
            print(
                f"{'Trans Error (m)*':<20} | {tm:<10.3f} | {tmed:<10.3f} | {tmax:<10.3f}"
            )

            # --- Added Scale Metric ---
            sm, smed, smax = stats(scales)
            print(f"{'Step Scale':<20} | {sm:<10.3f} | {smed:<10.3f} | {smax:<10.3f}")

            print("\n* Translation error is Relative Step Error (Scale Corrected)")
            print(
                "  Measures accuracy of the PnP step direction (ignoring map scale drift)."
            )

        print("=" * 65)


# ==========================================
# Alignment Logic Explanation
# ==========================================
# Monocular VO lacks a metric scale. The estimated trajectory shape might be
# correct, but the size will differ from Ground Truth (scale drift).
# To compare 'Translation Accuracy', we usually align the trajectories.
# In the console output above, I calculate a simple 'Scale Ratio'
# (Distance_GT / Distance_Est) and multiply the estimated position by it
# to give you a rough idea of the drift-corrected error.

if __name__ == "__main__":
    # Update paths
    IMG_DIR = "data/kitti/05/image_0"
    POSE_FILE = "data/kitti/poses/05.txt"
    OUT_DIR = "src/testbench/estimation_results"

    tb = PoseTestbench(IMG_DIR, POSE_FILE, OUT_DIR, start_frame=0, end_frame=500)
    tb.run()
