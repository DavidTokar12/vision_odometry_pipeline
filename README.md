# Vision Odometry Pipeline

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

### Import Datasets
1. Download the compressed dataset archive from the course website (https://rpg.ifi.uzh.ch/teaching.html#VAMR) and extract them. 
2. In `data/`, find the directory of the dataset you want to use and create a new folder called `images` in it.
3. Copy the images from the extracted archive into the newly created folder.
   1. For the `parking` dataset, copy the contents from the archive's `images/` directory.
   2. For the `kitti` dataset, copy the contents of the archive's `05/images_0` directory.
   3. For the `malaga` dataset, copy the contents of the archive's `Images/` directory. The 

### Shortcomings
The following point may be problematic for a more general VO pipeline as they are currently implemented in a very basic way:
- Parameters for the PNP/Ransac and the KTL functions were determined by trial and error to work with the provided datasets.

## Project Tools

### Package Management: uv

This project uses **[uv](https://github.com/astral-sh/uv)** for fast, reliable package management and dependency resolution.

To install dependencies:

```bash
uv sync
```

To add a new package:

```bash
uv add package_name
```

### Linting & Formatting: Ruff

This project uses **[Ruff](https://github.com/astral-sh/ruff)** for fast Python linting and code formatting.

Check code quality:

```bash
ruff check .
```

Auto-fix issues:

```bash
ruff check . --fix
```

Format code:

```bash
ruff format .
```

Check and format in one command:

```bash
ruff check . --fix && ruff format .
```

## Running the Project

The main entry point is `src/vision_odometry_pipeline/main.py`:

```bash
python src/vision_odometry_pipeline/main.py
```

For interactive development, you can also use the Jupyter notebook at `src/vision_odometry_pipeline/main.ipynb`.

## Development

When working in the dev container:

1. Install dev dependencies:
   ```bash
   uv sync --all-extras
   ```

2. Run linting and formatting before committing:
   ```bash
   ruff check . --fix && ruff format .
   ```

3. Run tests:
   ```bash
   pytest
   ```
