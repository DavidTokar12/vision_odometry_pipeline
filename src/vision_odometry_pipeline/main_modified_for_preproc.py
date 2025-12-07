from __future__ import annotations

import os

import cv2
import numpy as np

# Adjust this import based on where you saved the utility function
from vision_odometry_pipeline.utils.camera_utils import create_undistorted_maps
from vision_odometry_pipeline.vo_recorder import VoRecorder
from vision_odometry_pipeline.vo_runner import VoRunner


def main():
    parking_path = "/workspaces/vision_odometry_pipeline/data/parking"
    debug_output = "/workspaces/vision_odometry_pipeline/debug_output/main/"

    assert os.path.exists(parking_path), f"Dataset path not found: {parking_path}"
    os.makedirs(debug_output, exist_ok=True)

    # --- 1. Load Calibration Data (K and D) ---
    k_path = os.path.join(parking_path, "K.txt")
    assert os.path.exists(k_path), "K.txt not found in dataset folder"
    K = np.loadtxt(k_path, delimiter=",", usecols=(0, 1, 2))

    # Try to load Distortion Coefficients (D)
    d_path = os.path.join(parking_path, "D.txt")
    if os.path.exists(d_path):
        D = np.loadtxt(d_path, delimiter=",")
    else:
        print("Warning: D.txt not found. Assuming zero distortion.")
        D = np.zeros(5)  # Standard 5-param distortion model

    print(f"Loaded Camera Matrix K:\n{K}")
    print(f"Loaded Distortion D: {D}")

    # --- 2. Setup Image Paths ---
    images_dir = os.path.join(parking_path, "images")
    if not os.path.exists(images_dir):
        images_dir = parking_path

    # --- 3. Compute Undistortion Maps (One-Time Setup) ---
    # We need to read the first image to get dimensions (H, W)
    first_image_path = os.path.join(images_dir, "img_00000.png")
    if not os.path.exists(first_image_path):
        first_image_path = os.path.join(images_dir, "000000.png")

    first_img = cv2.imread(first_image_path)
    if first_img is None:
        raise FileNotFoundError(
            f"Could not load first image at {first_image_path} to initialize maps."
        )

    h, w = first_img.shape[:2]
    print(f"Image Resolution: {w}x{h}. Generating undistortion maps...")

    # This function generates the look-up tables and the NEW optimal K
    map_x, map_y, roi, new_K = create_undistorted_maps(K, D, (h, w))

    # --- 4. Initialize Runner with PRE-COMPUTED Maps ---
    print(f"Initializing VO Runner... Debug output: {debug_output}")

    # IMPORTANT: We pass 'new_K' because that is the geometry the VO algorithms will see
    # after the ImagePreprocessingStep runs. We also pass the maps.
    runner = VoRunner(
        K=new_K,
        undistortion_maps=(map_x, map_y, roi),
        debug=True,
        debug_output=debug_output,
    )

    recorder = VoRecorder(output_path=f"{debug_output}/out.mp4")

    # --- 5. Main Processing Loop ---
    last_frame = 598

    for i in range(last_frame + 1):
        image_name = f"img_{i:05d}.png"
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir, f"{i:06d}.png")

        if not os.path.exists(image_path):
            print(f"Warning: Frame {i} not found at {image_path}. Skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image {image_path}")
            continue

        print(f"Processing Frame {i:04d}...", end="\r")

        try:
            # Runner uses the pre-computed maps internally in ImagePreprocessingStep
            state = runner.process_frame(image)
            recorder.update(state=state, full_trajectory=runner.get_trajectory())
        except Exception as e:
            print(f"\nCritical Failure at Frame {i}: {e}")
            import traceback

            traceback.print_exc()
            break

    print("\nProcessing Complete.")
    recorder.close()
    print(f"Video saved to {debug_output}/out.mp4")


if __name__ == "__main__":
    main()
