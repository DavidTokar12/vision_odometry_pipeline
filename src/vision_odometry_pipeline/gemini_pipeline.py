import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1. Configuration (Winning Values)
# ==============================================================================


class Config:
    # Paths
    DATA_DIR = "data/kitti05"
    IMG_DIR = os.path.join(DATA_DIR, "images")
    K_FILE = os.path.join(DATA_DIR, "K.txt")
    POSES_FILE = os.path.join(DATA_DIR, "poses.txt")

    # Feature Detection (Shi-Tomasi)
    NUM_FEATURES = 4000
    QUALITY_LEVEL = 0.001
    MIN_DISTANCE = 5
    BLOCK_SIZE = 3

    # KLT Tracker
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # Mapping
    MIN_TRIANGULATION_BASELINE = 0.15


# ==============================================================================
# 2. Data Loading & Helpers
# ==============================================================================


def load_calib(filepath):
    """Robustly loads the K (calibration) matrix."""
    with open(filepath) as f:
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split(",") if x.strip()]
            data.append(row)
    return np.array(data, dtype=np.float32)


def load_poses(filepath):
    """Loads Ground Truth poses (T_WC: Camera-to-World)."""
    poses = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            T = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def inv_pose(T):
    """
    Inverts a 3x4 or 4x4 pose matrix.
    Input: T (3x4 or 4x4) [R|t]
    Output: T_inv (3x4) [R'|-R't]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return np.hstack((R_inv, t_inv.reshape(3, 1)))


# ==============================================================================
# 3. Visual Odometry Pipeline
# ==============================================================================


class VisualOdometry:
    def __init__(self, K):
        self.K = K
        # State: Always store T_CW (World-to-Camera) internally for projection
        self.cur_pose_cw = None
        self.prev_img = None

        self.px_ref = np.empty((0, 2), dtype=np.float32)
        self.landmarks = np.empty((0, 3), dtype=np.float32)
        self.candidates = []

    def feature_detection(self, img, mask=None):
        pts = cv2.goodFeaturesToTrack(
            img,
            mask=mask,
            maxCorners=Config.NUM_FEATURES,
            qualityLevel=Config.QUALITY_LEVEL,
            minDistance=Config.MIN_DISTANCE,
            blockSize=Config.BLOCK_SIZE,
        )
        if pts is not None:
            return pts.reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float32)

    def triangulation(self, T1_cw, T2_cw, pts1, pts2):
        """Triangulates using T_CW (Projection matrices)."""
        pts1_t = pts1.T
        pts2_t = pts2.T
        P1 = self.K @ T1_cw
        P2 = self.K @ T2_cw
        pts4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
        return (pts4d[:3] / pts4d[3]).T

    def process_initialization(self, img0, img1, gt_pos0, gt_pos1):
        """
        Robust Initialization:
        1. Calculates visual pose (Essential Matrix) to ensure rays intersect.
        2. Uses GT only to scale the result to meters.
        """
        # 1. Detect & Track
        p0 = self.feature_detection(img0)
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **Config.LK_PARAMS
        )

        # Filter Good Points
        good_mask = status.flatten() == 1
        p0_good = p0[good_mask]
        p1_good = p1[good_mask]

        if len(p0_good) < 100:
            print("  [Init] Failed: Not enough features.")
            return False

        # 2. Compute Relative Pose Visually (The Truth for the Camera)
        # This calculates the rotation and direction from pixels, ignoring potentially bad GT orientation
        E, mask = cv2.findEssentialMat(
            p1_good, p0_good, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        _, R, t, mask = cv2.recoverPose(E, p1_good, p0_good, self.K)

        # 3. Calculate Scale from Ground Truth
        # Distance = norm(Pos1 - Pos0)
        baseline = np.linalg.norm(gt_pos1 - gt_pos0)

        if baseline < 0.5:  # SAFETY CHECK
            print(
                f"  [Init] Failed: Baseline too small ({baseline:.2f}m). Car hasn't moved!"
            )
            return False

        # Apply scale to the unit translation vector
        t = t * baseline

        # 4. Create Projection Matrices for Triangulation
        # Camera 0 is at World Origin (Identity)
        T0_cw = np.eye(4)[:3, :]
        # Camera 1 is relative to Camera 0
        T1_cw = np.hstack((R, t))

        # 5. Triangulate
        landmarks = self.triangulation(T0_cw, T1_cw, p0_good, p1_good)

        # 6. Sanity Check Triangulation
        # Remove points that are 0, behind camera, or too far
        mask_tri = mask.flatten() == 255
        landmarks = landmarks[mask_tri]
        p1_good = p1_good[mask_tri]

        # Check Z (Depth)
        # Transform points to Camera 1 frame to check depth
        pts_cam1 = (R @ landmarks.T).T + t.T
        valid_depth = (pts_cam1[:, 2] > 0.5) & (pts_cam1[:, 2] < 500)

        self.landmarks = landmarks[valid_depth]
        self.px_ref = p1_good[valid_depth]

        # Initialize State
        self.cur_pose_cw = T1_cw
        self.prev_img = img1
        self._detect_new_candidates(img1, T1_cw)  # Fill up features

        print(
            f"  [Init] Success! {len(self.landmarks)} landmarks created. Baseline: {baseline:.2f}m"
        )
        return True

    def process_frame(self, img):
        """
        Returns: T_WC (Camera Position in World) for plotting.
        """
        # 1. Track
        p0 = self.px_ref
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, p0, None, **Config.LK_PARAMS
        )

        good_mask = status.flatten() == 1
        p1_good = p1[good_mask]
        landmarks_good = self.landmarks[good_mask]

        if len(p1_good) < 20:
            print(f"ERROR: len(p1_good) = {len(p1_good)}")
            return None

        # 2. Pose Estimation (PnP returns T_CW)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            landmarks_good,
            p1_good,
            self.K,
            None,
            iterationsCount=100,  # 100
            reprojectionError=2.0,  # 2.0
            confidence=0.99,
        )

        if not success or inliers is None:
            print(
                f"ERROR: success (should be True) = {success}, Inliers (should not be None) = {inliers}"
            )
            return None

        R, _ = cv2.Rodrigues(rvec)
        T_cur_cw = np.hstack((R, tvec))
        self.cur_pose_cw = T_cur_cw

        # Refine map
        inliers = inliers.flatten()
        self.px_ref = p1_good[inliers]
        self.landmarks = landmarks_good[inliers]

        # 3. Mapping
        self._track_candidates(img)
        self._triangulate_candidates(T_cur_cw)

        if len(self.px_ref) < Config.NUM_FEATURES:
            self._detect_new_candidates(img, T_cur_cw)

        self.prev_img = img

        # Return T_WC (Camera Position) for Plotting
        return inv_pose(T_cur_cw)

    def _track_candidates(self, cur_img):
        if not self.candidates:
            return
        p0 = np.array([c["cur_px"] for c in self.candidates], dtype=np.float32)
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img, cur_img, p0, None, **Config.LK_PARAMS
        )

        status = status.flatten()
        new_candidates = []
        for i, matched in enumerate(status):
            if matched:
                self.candidates[i]["cur_px"] = p1[i]
                new_candidates.append(self.candidates[i])
        self.candidates = new_candidates

    def _triangulate_candidates(self, cur_pose_cw):
        # Calculate Camera Centers (C = -R^T * t)
        # Since input is T_CW, this formula gives Camera Position in World
        def get_center(T_cw):
            return -T_cw[:3, :3].T @ T_cw[:3, 3]

        C_curr = get_center(cur_pose_cw)

        remaining_candidates = []
        for cand in self.candidates:
            first_pose_cw = cand["first_pose"]
            C_first = get_center(first_pose_cw)

            baseline = np.linalg.norm(C_curr - C_first)

            if baseline > Config.MIN_TRIANGULATION_BASELINE:
                pt1 = np.array([cand["first_px"]], dtype=np.float32)
                pt2 = np.array([cand["cur_px"]], dtype=np.float32)

                # Triangulate using T_CW matrices
                pt3d = self.triangulation(first_pose_cw, cur_pose_cw, pt1, pt2)

                # Cheirality: Project to current camera Z
                pt3d_cam = (cur_pose_cw[:3, :3] @ pt3d.T).T + cur_pose_cw[:3, 3]

                if pt3d_cam[0, 2] > 0:
                    self.landmarks = np.vstack((self.landmarks, pt3d))
                    self.px_ref = np.vstack((self.px_ref, pt2))
            else:
                remaining_candidates.append(cand)
        self.candidates = remaining_candidates

    def _detect_new_candidates(self, img, cur_pose_cw):
        mask = np.full(img.shape, 255, dtype=np.uint8)
        for pt in self.px_ref:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)
        for cand in self.candidates:
            pt = cand["cur_px"]
            cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)

        new_pts = self.feature_detection(img, mask)
        for pt in new_pts:
            # Store T_CW in the candidate
            self.candidates.append(
                {"first_pose": cur_pose_cw, "first_px": pt, "cur_px": pt}
            )


# ==============================================================================
# 4. Main Execution
# ==============================================================================


def main():
    try:
        K = load_calib(Config.K_FILE)
        gt_poses = load_poses(Config.POSES_FILE)  # These are T_WC
        images = sorted(
            [
                img
                for img in os.listdir(Config.IMG_DIR)
                if img.endswith((".png", ".jpg"))
                and not img.endswith(("right.png", "right.jpg"))
            ]
        )
    except Exception as e:
        print(f"[Error] {e}")
        return

    print(f"Loaded {len(images)} images.")
    vo = VisualOdometry(K)

    # Init
    img0 = cv2.imread(os.path.join(Config.IMG_DIR, images[0]), cv2.IMREAD_GRAYSCALE)
    # We keep Frame 0 as the "Anchor" and try to match it with Frame 1, 2, 3...
    # until we find enough movement.
    img0 = cv2.imread(os.path.join(Config.IMG_DIR, images[0]), cv2.IMREAD_GRAYSCALE)
    gt_pos0 = gt_poses[0][:3, 3]  # XYZ position

    start_index = 0

    for i in range(1, len(images)):
        img_curr = cv2.imread(
            os.path.join(Config.IMG_DIR, images[i]), cv2.IMREAD_GRAYSCALE
        )
        gt_pos_curr = gt_poses[i][:3, 3]

        dist = np.linalg.norm(gt_pos_curr - gt_pos0)

        # Try to initialize if distance > 0.5 meters
        if dist > 0.5:
            print(f"Attempting initialization at Frame {i} (Dist: {dist:.2f}m)...")

            # Pass just the positions (XYZ), not the full pose matrices
            if vo.process_initialization(img0, img_curr, gt_pos0, gt_pos_curr):
                start_index = i
                break
        else:
            print(f"Skipping Frame {i}: Dist {dist:.2f}m (Too small)")

    if start_index == 0:
        print("Error: Could not initialize (Vehicle never moved?)")
        return

    # --- TRACKING LOOP ---
    print(f"\n--- TRACKING STARTED from Frame {start_index} ---")

    # Fill trajectory with static copies for the skipped frames (for plotting alignment)
    est_traj = [gt_pos0] * start_index
    gt_traj = [gt_poses[k][:3, 3] for k in range(start_index)]

    # Visuals
    traj_bg = np.zeros((800, 800, 3), dtype=np.uint8)

    # --- VIDEO WRITER SETUP ---
    # Calculate output dimensions based on the logic inside the loop
    # Logic: map width is 2x image height, stack is horizontal
    h, w = img0.shape[:2]
    map_width = h * 2
    total_width = w + map_width
    total_height = h

    fps = 20.0  # Adjust if video plays too fast/slow
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    debug_output = "debug_output/gemini"
    os.makedirs(debug_output, exist_ok=True)

    video_out = cv2.VideoWriter(
        f"{debug_output}/vo_gemini_live.mp4", fourcc, fps, (total_width, total_height)
    )
    # --------------------------

    print("Running VO Loop...")
    for i in range(start_index + 1, len(images)):
        img = cv2.imread(os.path.join(Config.IMG_DIR, images[i]), cv2.IMREAD_GRAYSCALE)

        # process_frame now returns T_WC (Camera Position)
        T_WC_est = vo.process_frame(img)

        if T_WC_est is None:
            T_WC_est = np.zeros((4, 4))
            print(f"Tracking failed after {i} frames.")
            # break

        est_pos = T_WC_est[:3, 3]
        gt_pos = gt_poses[i][:3, 3]

        est_traj.append(est_pos)
        gt_traj.append(gt_pos)

        # --- Visualization ---
        img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for p in vo.px_ref:
            cv2.circle(img_disp, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        for c in vo.candidates:
            cv2.circle(
                img_disp, (int(c["cur_px"][0]), int(c["cur_px"][1])), 2, (0, 0, 255), -1
            )

        # Plotting logic
        scale = 1
        off_x = 400
        off_y = 700
        dx_est, dy_est = (
            int(est_pos[0] * scale) + off_x,
            int(off_y - est_pos[2] * scale),
        )
        dx_gt, dy_gt = int(gt_pos[0] * scale) + off_x, int(off_y - gt_pos[2] * scale)

        cv2.circle(traj_bg, (dx_est, dy_est), 1, (0, 255, 0), 1)
        cv2.circle(traj_bg, (dx_gt, dy_gt), 1, (0, 0, 255), 1)

        # Resize map to match image height (Height = img.shape[0])
        map_resized = cv2.resize(traj_bg, (img_disp.shape[0] * 2, img_disp.shape[0]))

        # Combine them
        combined_view = np.hstack((img_disp, map_resized))

        # Write to video
        if i % 3 == 0:
            video_out.write(combined_view)

    video_out.release()
    print("Video saved to 'vo_live.mp4'")

    # Final Plot
    est_traj = np.array(est_traj)
    gt_traj = np.array(gt_traj)
    plt.figure(figsize=(10, 10))
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], "r--", label="Ground Truth")
    plt.plot(est_traj[:, 0], est_traj[:, 2], "g-", label="VO Estimate")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.savefig(f"{debug_output}/vo_gemini_result.png")
    plt.show()


if __name__ == "__main__":
    main()
