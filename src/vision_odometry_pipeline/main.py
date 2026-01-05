from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from vision_odometry_pipeline.image_sequence import ImageSequence
from vision_odometry_pipeline.vo_configs import DatasetConfig
from vision_odometry_pipeline.vo_configs import load_config
from vision_odometry_pipeline.vo_recorder import VoRecorder
from vision_odometry_pipeline.vo_runner_process import VoRunnerProcess


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


DATASET_DEFAULTS = {
    "parking": {
        "source_dir": "parking",
        "image_dir": "images",
        "image_pattern": "img_{:05d}.png",
        "last_frame": 598,
        "ground_truth_path": "poses.txt",
        "configs": "parking",
    },
    "kitti": {
        "source_dir": "kitti",
        "image_dir": "05/image_0",
        "image_pattern": "{:06d}.png",
        "last_frame": 2760,
        "ground_truth_path": "poses/05.txt",
        "configs": "kitti",
    },
    "malaga": {
        "source_dir": "malaga-urban-dataset-extract-07",
        "image_dir": "malaga-urban-dataset-extract-07_rectified_800x600_Images",
        "image_pattern": "*_left.jpg",
        "last_frame": 2120,
        "ground_truth_path": None,
        "configs": "malaga",
    },
    "polibahn": {
        "source_dir": "polibahn_down",
        "image_dir": "images",
        "image_pattern": "img_{:05d}.png",
        "last_frame": 1243,
        "ground_truth_path": None,
        "configs": "antonios_phone",
    },
    "sonneggstrasse": {
        "source_dir": "sonneggstrasse",
        "image_dir": "images",
        "image_pattern": "img_{:05d}.png",
        "last_frame": 837,
        "ground_truth_path": None,
        "configs": "antonios_phone",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual Odometry Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="parking",
        choices=list(DATASET_DEFAULTS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "-f",
        "--first-frame",
        type=int,
        default=0,
        help="First frame to process",
    )
    parser.add_argument(
        "-l",
        "--last-frame",
        type=int,
        default=None,
        help="Last frame to process (None for dataset default)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/workspaces/vision_odometry_pipeline/debug_output",
        help="Output directory base path",
    )
    parser.add_argument(
        "-dp",
        "--data-path",
        type=str,
        default="/workspaces/vision_odometry_pipeline/data",
        help="Base path to dataset folders",
    )
    parser.add_argument(
        "-cp",
        "--config-path",
        type=str,
        default="/workspaces/vision_odometry_pipeline/src/vision_odometry_pipeline/configs",
        help="Base path to config JSON files",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        action="store_true",
        default=False,
        help="Plot ground truth trajectory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (VO runner debug output)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]

    dataset = DatasetConfig(
        name=args.dataset,
        data_path=os.path.join(args.data_path, defaults["source_dir"]),
        image_dir=defaults["image_dir"],
        image_pattern=defaults["image_pattern"],
        first_frame=args.first_frame,
        last_frame=args.last_frame
        if args.last_frame is not None
        else defaults["last_frame"],
        ground_truth_path=defaults["ground_truth_path"],
        debug_output=os.path.join(args.output, args.dataset),
    )

    config_path = os.path.join(
        args.config_path, defaults["configs"], f"{defaults['configs']}.json"
    )
    config = load_config(config_path, dataset)

    sequence = ImageSequence(config)

    first_image = sequence.peek_image()
    if first_image is None:
        raise RuntimeError("Could not load first image")

    video_path = os.path.join(config.dataset.debug_output, "out.mp4")

    timing_history: list[dict[str, float]] = []

    logger.info("Initializing VO Runner Process...")

    with VoRunnerProcess(
        config=config,
        image_shape=first_image.shape,
        image_dtype=first_image.dtype,
        initial_frame=args.first_frame,
        debug=args.debug,
    ) as runner:
        recorder = VoRecorder(
            output_path=video_path, plot_ground_truth=args.ground_truth
        )

        if args.ground_truth and config.ground_truth is not None:
            recorder.set_ground_truth(config.ground_truth)
            logger.info("Ground truth loaded: %d poses", len(config.ground_truth))

        try:
            for frame_id, image in sequence:
                runner.submit_frame(frame_id, image)
                result = runner.get_result(timeout=30.0)

                if result is None:
                    logger.error("Timeout waiting for frame %d", frame_id)
                    break

                if result.step_timings:
                    timing_history.append(result.step_timings)

                recorder.update(
                    image=image,
                    result=result,
                    full_trajectory=result.trajectory,
                )
                print(
                    f"Processed Frame {result.frame_id:04d} in {result.processing_time_ms:.1f} ms",
                    end="\r",
                )
        except Exception as e:
            logger.exception(
                "Critical failure at frame %d: %s", sequence.current_idx, e
            )
            raise

    print()
    logger.info("Processing complete")

    recorder.close()
    logger.info("Video saved to %s", video_path)

    if timing_history:
        df = pd.DataFrame(timing_history)
        stats = df.agg(["mean", "std", "min", "max"]).T
        stats.columns = ["mean_ms", "std_ms", "min_ms", "max_ms"]
        stats["fps"] = 1000 / stats["mean_ms"]
        stats = stats.round(2)

        stats_path = os.path.join(config.dataset.debug_output, "performance.csv")

        stats.to_csv(stats_path)
        logger.info("Performance stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
