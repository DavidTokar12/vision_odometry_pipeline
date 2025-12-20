import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1. Configuration (Winning Values)
# ==============================================================================


class Config:
    # Paths
    DATA_DIR = "data/parking"
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

    def process_initialization(self, img0, img1, T0_wc, T1_wc):
        """
        Input: GT Poses T_WC (Camera Position).
        Action: Invert them to T_CW for triangulation logic.
        """
        # 1. Convert GT (Position) to Projection Matrix (Extrinsics)
        T0_cw = inv_pose(T0_wc)
        T1_cw = inv_pose(T1_wc)

        # 2. Detect & Track
        p0 = self.feature_detection(img0)
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **Config.LK_PARAMS
        )

        good_mask = status.flatten() == 1
        p0_good = p0[good_mask]
        p1_good = p1[good_mask]

        # 3. Triangulate (using T_CW)
        landmarks = self.triangulation(T0_cw, T1_cw, p0_good, p1_good)

        # Init State
        self.px_ref = p1_good
        self.landmarks = landmarks
        self.cur_pose_cw = T1_cw  # Store T_CW
        self.prev_img = img1

        self._detect_new_candidates(img1, T1_cw)
        print(f"Initialized with {len(self.landmarks)} landmarks.")

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
            return None

        # 2. Pose Estimation (PnP returns T_CW)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            landmarks_good,
            p1_good,
            self.K,
            None,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
        )

        if not success or inliers is None:
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
            [img for img in os.listdir(Config.IMG_DIR) if img.endswith(".png")]
        )
    except Exception as e:
        print(f"[Error] {e}")
        return

    print(f"Loaded {len(images)} images.")
    vo = VisualOdometry(K)

    # Init
    img0 = cv2.imread(os.path.join(Config.IMG_DIR, images[0]), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(Config.IMG_DIR, images[1]), cv2.IMREAD_GRAYSCALE)

    # T0, T1 are GT (T_WC). Passed directly to process_initialization, which now inverts them.
    vo.process_initialization(img0, img1, gt_poses[0], gt_poses[1])

    # Store Trajectories (Position only)
    est_traj = [gt_poses[0][:3, 3], gt_poses[1][:3, 3]]
    gt_traj = [gt_poses[0][:3, 3], gt_poses[1][:3, 3]]

    # Visuals
    # cv2.namedWindow("VO Pipeline", cv2.WINDOW_NORMAL)
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
    video_out = cv2.VideoWriter(
        "debug_output/gemini/vo_live.mp4", fourcc, fps, (total_width, total_height)
    )
    # --------------------------

    print("Running VO Loop...")
    for i in range(2, len(images)):
        img = cv2.imread(os.path.join(Config.IMG_DIR, images[i]), cv2.IMREAD_GRAYSCALE)

        # process_frame now returns T_WC (Camera Position)
        T_WC_est = vo.process_frame(img)

        if T_WC_est is None:
            print("Tracking failed.")
            break

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

        # ... (previous plotting logic) ...

        # Resize map to match image height (Height = img.shape[0])
        # Your logic: Width = Height * 2
        map_resized = cv2.resize(traj_bg, (img_disp.shape[0] * 2, img_disp.shape[0]))

        # Combine them
        combined_view = np.hstack((img_disp, map_resized))

        # Write to video
        if i % 3 == 0:
            video_out.write(combined_view)

        # Show on screen
        # cv2.imshow("VO Pipeline", combined_view)
        # if cv2.waitKey(1) == 27:
        # break

    # cv2.destroyAllWindows()

    video_out.release()
    print("Video saved to 'debug_output/gemini/vo_live.mp4'")

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
    plt.savefig("debug_output/gemini/vo_result.png")
    plt.show()


if __name__ == "__main__":
    main()
