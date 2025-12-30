from __future__ import annotations

import os

from vision_odometry_pipeline.image_sequence import ImageSequence
from vision_odometry_pipeline.vo_recorder import VoRecorder
from vision_odometry_pipeline.vo_runner import VoRunner


def main():
    # 0: Parking, 1: KITTI, 2: Malaga
    dataset_selection = 1
    first_frame = 0
    last_frame = 1000

    # Enable/Disable Ground Truth Plotting
    plot_ground_truth = False

    # Initialize DataLoader
    sequence = ImageSequence(
        dataset_id=dataset_selection, first_frame=first_frame, last_frame=last_frame
    )

    print("Initializing VO Runner...")
    runner = VoRunner(
        K=sequence.K,
        D=sequence.D,
        initial_frame=first_frame,
        debug=True,
        debug_output=sequence.debug_output,
    )

    # Initialize Recorder
    video_path = os.path.join(sequence.debug_output, "out.mp4")
    recorder = VoRecorder(output_path=video_path, plot_ground_truth=plot_ground_truth)

    # Set ground truth if available and plotting is enabled
    if plot_ground_truth and sequence.ground_truth is not None:
        recorder.set_ground_truth(sequence.ground_truth)
        print(f"Ground truth loaded: {len(sequence.ground_truth)} poses")

    # Main Loop
    while not sequence.is_finished:
        frame_id = sequence.current_idx

        image = sequence.get_image()
        if image is None:
            print("Stream ended or image load failure.")
            break

        try:
            state = runner.process_frame(image)

            # If ground truth is needed for recording/plotting
            # you can access loader.ground_truth here
            recorder.update(state=state, full_trajectory=runner.get_trajectory())
            print(
                f"Processed Frame {frame_id:04d} in {runner.last_processing_time:.1f} ms",
                end="\r",
            )

        except Exception as e:
            print(f"\nCritical Failure at Frame {frame_id}: {e}")
            break

    print("\nProcessing Complete.")
    recorder.close()
    recorder.compress()
    print(f"Video saved to {video_path}")

    runner.print_diagnostics()


if __name__ == "__main__":
    main()
