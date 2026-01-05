# Vision Odometry Pipeline

## Overview

A monocular visual odometry pipeline implementing two-view initialization, KLT optical flow tracking, RANSAC-PnP pose estimation, landmark triangulation, and local bundle adjustment. Achieves 15-20 FPS on standard datasets (KITTI, Malaga, Parking) and custom recordings. Developed for the Vision Algorithms for Mobile Robotics course at UZH, taught by Prof. Dr. Davide Scaramuzza.

## Performance

Screencasts of the pipeline in action can be found on [YouTube](https://www.youtube.com/playlist?list=PL-waHff4z-BM604fTs38r3McEt11LEM0O).
For detailed explanations and in-depth analysis of the performance including hardware specifications, please refer to the project [report](Report.pdf).

## Setup Instructions

### Prerequisites

Before getting started, ensure you have the following installed:

1. **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
2. **VS Code Dev Containers Extension** - Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension from the VS Code marketplace

### Quick Start

1. Clone the repository and open it in VS Code
2. When prompted, click **"Reopen in Container"** or:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Search for and select **"Dev Containers: Reopen in Container"**
3. Wait for the container to build and open
4. Verify the correct interpreter (`.venv`) is selected

The development environment will be ready once the container is open.

### Dataset Setup

1. Download the compressed dataset archive from the [course website](https://rpg.ifi.uzh.ch/teaching.html#VAMR) and from [here](https://u.ethz.ch/0bMXU).
2. Create a `data/` directory in the project root.
3. Extract the dataset archive directly into `data/`. Keep the default folder structure from the archive unchanged.

Your directory should look like:
```
vision_odometry_pipeline/
├── data/
│   ├── parking/
│   ├── kitti/
│   ├── malaga-urban-dataset-extract-07/
│   ├── polibahn_down/
│   └── sonneggstrasse/
├── src/
└── ...
```

## Running the Pipeline

Run the pipeline using `uv run`:
```bash
uv run python ./src/vision_odometry_pipeline/main.py -d kitti
```

### Command Line Arguments

| Argument         | Short | Default                        | Description                                                                             |
| ---------------- | ----- | ------------------------------ | --------------------------------------------------------------------------------------- |
| `--dataset`      | `-d`  | `parking`                      | Dataset to process. Choices: `parking`, `kitti`, `malaga`, `polibahn`, `sonneggstrasse` |
| `--first-frame`  | `-f`  | `0`                            | First frame to process                                                                  |
| `--last-frame`   | `-l`  | `None`                         | Last frame to process (uses dataset default if not specified)                           |
| `--output`       | `-o`  | `/workspaces/.../debug_output` | Output directory base path                                                              |
| `--data-path`    | `-dp` | `/workspaces/.../data`         | Base path to dataset folders                                                            |
| `--config-path`  | `-cp` | `/workspaces/.../configs`      | Base path to config JSON files                                                          |
| `--ground-truth` | `-g`  | `False`                        | Plot ground truth trajectory                                                            |
| `--debug`        |       | `False`                        | Enable debug mode (saves intermediate outputs)                                          |

### Examples
```bash
# Run on parking dataset (default)
uv run python ./src/vision_odometry_pipeline/main.py

# Run on KITTI dataset
uv run python ./src/vision_odometry_pipeline/main.py -d kitti

# Run on KITTI with ground truth overlay
uv run python ./src/vision_odometry_pipeline/main.py -d kitti -g

# Process only frames 100-200
uv run python ./src/vision_odometry_pipeline/main.py -d kitti -f 100 -l 200

# Enable debug output
uv run python ./src/vision_odometry_pipeline/main.py -d parking --debug
```

## Package Management with uv

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable package management.

### Install Dependencies
```bash
uv sync
```

### Install with Dev Dependencies
```bash
uv sync --all-extras
```

### Add a New Package
```bash
uv add package_name
```

### Run Any Python Command
```bash
uv run python <script.py>
uv run pytest
uv run ruff check .
```

## Linting & Formatting with Ruff

This project uses **[Ruff](https://github.com/astral-sh/ruff)** for Python linting and formatting.
```bash
# Check code quality
uv run ruff check .

# Format code
uv run ruff format .
```