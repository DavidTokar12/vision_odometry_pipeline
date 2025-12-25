from __future__ import annotations

import os

from vision_odometry_pipeline.image_sequence import ImageSequence
from vision_odometry_pipeline.vo_recorder import VoRecorder
from vision_odometry_pipeline.vo_runner import VoRunner


def main():
    # 0: Parking, 1: KITTI, 2: Malaga
    dataset_selection = 0
    image_range = 100

    # Initialize DataLoader
    sequence = ImageSequence(dataset_id=dataset_selection, last_frame=image_range)

    print("Initializing VO Runner...")
    runner = VoRunner(
        K=sequence.K, D=sequence.D, debug=True, debug_output=sequence.debug_output
    )

    # Initialize Recorder
    video_path = os.path.join(sequence.debug_output, "out.mp4")
    recorder = VoRecorder(output_path=video_path)

    # Main Loop
    while not sequence.is_finished:
        frame_id = sequence.current_idx
        print(f"Processing Frame {frame_id:04d}...", end="\r")

        image = sequence.get_image()
        if image is None:
            print("Stream ended or image load failure.")
            break

        try:
            state = runner.process_frame(image)

            # If ground truth is needed for recording/plotting
            # you can access loader.ground_truth here
            recorder.update(state=state, full_trajectory=runner.get_trajectory())

        except Exception as e:
            print(f"\nCritical Failure at Frame {frame_id}: {e}")
            break

    print("\nProcessing Complete.")
    recorder.close()
    recorder.compress()
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
