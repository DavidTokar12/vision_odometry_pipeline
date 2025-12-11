from __future__ import annotations

import os

import cv2
import numpy as np

from vision_odometry_pipeline.vo_recorder import VoRecorder
from vision_odometry_pipeline.vo_runner import VoRunner


def main():
    # 1: parking, 2: kitti, 3: malaga, 4: own video
    selection = 2
    if selection == 1:
        data_path = "/workspaces/vision_odometry_pipeline/data/parking"
        debug_output = "/workspaces/vision_odometry_pipeline/debug_output/main_parking/"
    elif selection == 2:
        data_path = "/workspaces/vision_odometry_pipeline/data/kitti05"
        debug_output = "/workspaces/vision_odometry_pipeline/debug_output/main_kitti/"

    assert os.path.exists(data_path), f"Dataset path not found: {data_path}"

    os.makedirs(debug_output, exist_ok=True)

    k_path = os.path.join(data_path, "K.txt")
    assert os.path.exists(k_path), "K.txt not found in dataset folder"
    K = np.loadtxt(k_path, delimiter=",", usecols=(0, 1, 2))

    print(f"Loaded Camera Matrix K:\n{K}")

    d_path = os.path.join(data_path, "D.txt")
    if os.path.exists(d_path):
        D = np.loadtxt(d_path, delimiter=",")
        print(f"Loaded Distortion Vector:\n{D}")
    else:
        print("Warning: D.txt not found. Assuming zero distortion.")
        D = np.zeros(5)

    print(f"Initializing VO Runner... Debug output: {debug_output}")

    runner = VoRunner(K=K, D=D, debug=True, debug_output=debug_output)
    recorder = VoRecorder(output_path=f"{debug_output}/out.mp4")

    last_frame = 200

    images_dir = os.path.join(data_path, "images")
    if not os.path.exists(images_dir):
        images_dir = data_path

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
            state = runner.process_frame(image)
            recorder.update(state=state, full_trajectory=runner.get_trajectory())
        except Exception as e:
            print(f"\nCritical Failure at Frame {i}: {e}")
            break

    print("\nProcessing Complete.")

    recorder.close()
    recorder.compress()
    print(f"Video saved to {debug_output}/out.mp4")


if __name__ == "__main__":
    main()
