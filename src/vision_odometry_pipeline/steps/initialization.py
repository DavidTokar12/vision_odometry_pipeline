from dataclasses import dataclass

import cv2
import numpy as np


# --- Configuration ---
@dataclass
class InitializationConfig:
    lk_win_size: tuple[int, int] = (15, 15)
    lk_max_level: int = 3
    fb_max_dist: float = 1.0
    ransac_threshold: float = 1.0
    ransac_prob: float = 0.999
    min_buffer_size: int = 2
    min_inliers: int = 8


class InitializationStep:
    def __init__(self, config: InitializationConfig = InitializationConfig()):
        self.config = config

    def process(
        self, state: VoState, debug: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        # 1. Validation
        if not state.frame_buffer:
            return np.zeros((100, 100), dtype=np.uint8), None

        # Get the latest image for return consistency
        current_image = state.frame_buffer[-1]

        # Check if we have enough data to initialize
        if len(state.frame_buffer) < self.config.min_buffer_size:
            return current_image, None

        # 2. Setup
        # We assume state.K matches the resolution/distortion of the images in frame_buffer
        K = np.array(state.calibration_matrix, dtype=np.float32)

        # Ensure images are grayscale for Feature Detectors
        # We create a local list of grayscale views to avoid modifying state.frame_buffer in place
        gray_frames = [
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            for img in state.frame_buffer
        ]

        # 3. Feature Detection (SIFT on the FIRST frame of the buffer)
        img0 = gray_frames[0]
        sift = cv2.SIFT_create()
        kp0 = sift.detect(img0, None)

        if not kp0:
            print("Init: No keypoints found.")
            return current_image, None

        # Format points: (N, 1, 2)
        p_initial = np.float32([kp.pt for kp in kp0]).reshape(-1, 1, 2)
        current_pts = p_initial.copy()

        # 4. Track Features through the buffer (Optical Flow)
        prev_img = img0

        for i in range(1, len(gray_frames)):
            curr_img = gray_frames[i]

            # Helper function tracks from prev -> curr
            p1, good_mask = self._track_features_bidirectional(
                prev_img, curr_img, current_pts
            )

            # Keep only the points that survived tracking
            current_pts = p1[good_mask].reshape(-1, 1, 2)
            p_initial = p_initial[good_mask].reshape(-1, 1, 2)
            prev_img = curr_img

            if len(current_pts) < self.config.min_inliers:
                print("Init: Lost too many features during tracking.")
                return current_image, None

        # 5. Pose Estimation (Essential Matrix)
        pts1 = p_initial.reshape(-1, 2)
        pts2 = current_pts.reshape(-1, 2)

        # Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold,
        )

        if E is None:
            return current_image, None

        # Select RANSAC inliers
        mask = mask.ravel()
        pts1 = pts1[mask == 1]
        pts2 = pts2[mask == 1]

        # Recover Pose (R, t)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # Filter Chirality (keep points in front of camera)
        pts1 = pts1[mask_pose.ravel() == 255]
        pts2 = pts2[mask_pose.ravel() == 255]

        if len(pts1) < self.config.min_inliers:
            print("Init: Not enough valid points after pose recovery.")
            return current_image, None

        # 6. Triangulation (Get 3D Landmarks)
        pts3D = self._triangulate(R, t, pts1, pts2, K)

        # 7. Update State
        state.R = R
        state.t = t
        T_current = np.eye(4, dtype=np.float64)
        T_current[:3, :3] = R
        T_current[:3, 3] = t.ravel()
        state.pose = T_current
        state.landmarks = pts3D
        state.X = pts3D
        state.C = pts2
        state.F = pts1 #first observation pixel coordinates
        state.initial_keypoints = pts1
        state.current_keypoints = pts2
        state.is_initialized = True

        print(f"Init Success: {len(pts3D)} landmarks.")

        # 8. Debug Visualization
        debug_img = None
        if debug:
            # We visualize on the FIRST frame to show the flow vectors from start to end
            debug_img = self._create_debug_vis(state.frame_buffer[0], pts1, pts2)

        return current_image, debug_img

    # --- Internal Helpers ---

    def _track_features_bidirectional(self, img0, img1, p0):
        lk_params = dict(
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        # Backward flow
        p0r, st2, err2 = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        # Check consistency (L-infinity norm)
        dist = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good_mask = (
            (st1.flatten() == 1)
            & (st2.flatten() == 1)
            & (dist < self.config.fb_max_dist)
        )

        return p1, good_mask

    def _triangulate(self, R, t, pts1, pts2, K):
        # Projection Matrix 1 (World Origin)
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        # Projection Matrix 2 (Current Pose)
        P2 = K @ np.hstack((R, t))

        pts_homo = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        return (pts_homo / pts_homo[3])[:3].T

    def _create_debug_vis(self, img_bg, pts_start, pts_end):
        """Draws vectors from initial position to current position."""
        # Ensure we have a writable color image
        vis = (
            cv2.cvtColor(img_bg, cv2.COLOR_GRAY2BGR)
            if img_bg.ndim == 2
            else img_bg.copy()
        )

        for pt_s, pt_e in zip(pts_start, pts_end):
            x0, y0 = map(int, pt_s.ravel())
            x1, y1 = map(int, pt_e.ravel())

            cv2.line(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)  # Green path
            cv2.circle(vis, (x0, y0), 2, (0, 0, 255), -1)  # Red start
            cv2.circle(vis, (x1, y1), 3, (255, 0, 0), -1)  # Blue end

        return vis
