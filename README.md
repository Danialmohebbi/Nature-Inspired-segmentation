# Nature-Inspired Image Segmentation

This repository multilevel image segmentation experiments using classical methods and nature inspired optimization algorithms, evaluates the reconstructed images, and saves the measurements as CSV files.

This README serves as user documentation via the installation, command, data, results, and troubleshooting sections and the programming documentation via the project layout, architecture, API, data-contract, and extension sections.

## What the project does

For every requested algorithm, segment count, trial, and image in a data split, the experiment runner:

1. loads the image as grayscale or RGB;
2. segments it using a classical or nature-inspired method;
3. calculates MSE, PSNR, FSIM, SSIM, and QILV;
4. records the fitness value, convergence history, runtime, and thresholds; and
5. appends the result to a CSV file in `results/`.

## Requirements

- Windows, Linux, or macOS
- Python 3.10 recommended
- Enough memory and CPU time for the selected experiment

The pinned packages in `requirements.txt` require Python 3.10 or newer. Python 3.10 is the safest choice for reproducing this project. PyTorch is used on the CPU by the metric code; a GPU is not required by the current configuration.

## Installation

Run these commands from the repository root—the directory containing this README and `requirements.txt`.

### Windows PowerShell

```powershell
cd "<path-to-the-project>"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Quick start

Start with one fast classical method, one segment count, one trial, and the five-image `train_batch0` split:

```powershell
python experiments/run.py --algos otsu --kvals 2 --trials 1 --fitness kapur --split train_batch0
```

The run creates these files:

```text
results/train_batch0_output_gray_kapur.csv
results/train_batch0_output_color_kapur.csv
```

One file may contain only its header if the selected split has no successful images of that mode.

## Command-line options

```text
python experiments/run.py \
  --algos <comma-separated names> \
  --kvals <comma-separated integers> \
  --trials <positive integer> \
  --fitness <fitness name> \
  --split <split filename without .txt>
```

Example with several methods and segment counts:

```powershell
python experiments/run.py --algos pso,woa,otsu,kmeans --kvals 2,4 --trials 3 --fitness kapur --split train_batch0
```

### Supported algorithms

| Name | Type |
| --- | --- |
| `ga` | Genetic Algorithm |
| `cmaes` | Covariance Matrix Adaptation Evolution Strategy |
| `pso` | Particle Swarm Optimization |
| `woa` | Whale Optimization Algorithm |
| `gwo` | Grey Wolf Optimizer |
| `zoa` | Zebra Optimization Algorithm |
| `woazoa` | WOA/ZOA hybrid |
| `otsu` | Recursive Otsu thresholding |
| `kmeans` | K-means clustering |

### Supported fitness functions

| Name | Description |
| --- | --- |
| `kapur` | Kapur entropy |
| `kapur_spatial` | Kapur entropy with a spatial penalty |
| `kapur_ssim` | Blend of normalized Kapur entropy and SSIM |

`--fitness` affects nature-inspired methods. It is still a required argument for `otsu` and `kmeans`, but for those classical methods it only contributes to the output filename; it does not change their segmentation.

### Available data splits

- `train` — 50 images: 25 grayscale and 25 color
- `train_batch0` through `train_batch9` — 5 images each

Pass the split name without the `.txt` extension.

## Adding images or splits

Put image files in `data/images/`. Then create or edit a file in `data/splits/`, with one image per line:

```text
example-gray.png,gray
example-color.jpg,color
```

Only `gray` and `color` are accepted. Filenames are resolved relative to `data/images/`; nested paths can be used if they are written into the split file.

## Results

Results are appended to:

```text
results/<split>_output_<mode>_<fitness>.csv
```

Each row contains:

- algorithm, image ID, mode, `k`, trial, and fitness-function label;
- MSE, PSNR, FSIM, SSIM, and QILV;
- objective fitness;
- convergence history;
- runtime in seconds; and
- thresholds or K-means centroids.

The output directory is resolved from the source files, so results are always written to the repository's own `results/` directory—even if the entry point is launched from another working directory or the repository is moved.

Existing CSV files are not replaced. Re-running the same command appends another set of rows, including duplicate combinations. Rename or move previous result files before a clean rerun.

## Performance configuration

The main settings are currently constants in the source code:

- `evaluation/runner.py`: `CPU_COUNT = -1` uses all available logical CPU cores.
- `segmentation/base_segmenter.py`: `DEFAULT_EPOCHS = 500`.
- `segmentation/base_segmenter.py`: `DEFAULT_POP_SIZE = 50`.
- `fitness/reconstruction.py`: metrics use the CPU.

## Project layout

The source tree is shown below. Generated `results/` files is intentionally excluded.

```text
.
|-- README.md                         combined user and programmer guide
|-- requirements.txt                  pinned direct Python dependencies
|-- Thesis.pdf                        thesis document
|-- algorithms/
|   `-- woazoa.py                     custom MEALPY WOAZOA optimizer
|-- data/
|   |-- loader.py                     path-independent image/split loading
|   |-- images/                       50 input images
|   `-- splits/
|       |-- train.txt                 complete 50-image split
|       `-- train_batch0.txt ... train_batch9.txt
|                                      ten five-image batches
|-- evaluation/
|   `-- runner.py                     task creation, parallel work, CSV writing
|-- experiments/
|   `-- run.py                        command-line entry point
|-- fitness/
|   |-- kapur.py                      Kapur entropy and reconstruction helper
|   |-- kapur_spatial.py              spatially penalized Kapur objective
|   |-- kapur_ssim.py                 Kapur/SSIM blended objective
|   `-- reconstruction.py             MSE, PSNR, SSIM, FSIM, and QILV metrics
`-- segmentation/
    |-- base_segmenter.py             algorithm registry and common dispatcher
    |-- classical.py                  Otsu, K-means, and threshold application
    `-- problem.py                    MEALPY optimization-problem adapter
```

## Programmer documentation

### Architecture and execution flow

```text
experiments/run.py
    -> evaluation.runner.run_experiment()
        -> data.loader.get_image_paths()
        -> joblib creates parallel run() tasks
            -> data.loader.load_image()
            -> segmentation.base_segmenter.run_segmentation()
                -> classical Otsu/K-means, or
                -> MEALPY optimizer + SegmentationProblem objective
            -> fitness.reconstruction.compute_all_metrics()
        -> evaluation.runner.add_row()
            -> <repository>/results/*.csv
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `experiments/run.py` | Parses CLI arguments, establishes the repository root on `sys.path`, and starts an experiment. |
| `evaluation/runner.py` | Builds tasks, uses all CPU cores by default, catches per-task failures, and appends result rows. |
| `data/loader.py` | Resolves repository-relative data paths, parses split rows, and loads grayscale or RGB arrays. |
| `segmentation/base_segmenter.py` | Registers supported algorithms and normalizes their outputs into a common result dictionary. |
| `segmentation/problem.py` | Adapts an image and selected objective to MEALPY's minimization interface. |
| `segmentation/classical.py` | Implements recursive Otsu, K-means, and intensity reconstruction. |
| `algorithms/woazoa.py` | Implements the custom exploration/exploitation update for WOAZOA. |
| `fitness/kapur*.py` | Implements the three optimizer objectives. Lower values are better because MEALPY is configured for minimization. |
| `fitness/reconstruction.py` | Converts images to tensors when needed and calculates the five reported quality metrics. |

### Core interfaces

| Interface | Purpose and return value |
| --- | --- |
| `get_image_paths(split)` | Returns `(Path, mode)` pairs from `data/splits/<split>.txt`. |
| `load_image(path, mode)` | Returns a `uint8` NumPy array in grayscale `(H, W)` or RGB `(H, W, 3)` form. |
| `run_segmentation(name, image, k, mode, fitness_fn, params)` | Dispatches an algorithm and returns `thresholds`, `segmented_image`, `fitness`, and `history`. |
| `SegmentationProblem(image, k, mode, fitness_fn)` | Defines MEALPY bounds and evaluates candidate threshold vectors. |
| `compute_all_metrics(original, reconstructed, mode)` | Returns a dictionary containing MSE, PSNR, SSIM, FSIM, and QILV. |
| `run_experiment(algorithms, k_values, n_trials, fitness_fn, split)` | Executes the full task product and appends successful rows to the mode-specific CSV files. |

### Data contracts

- Images are `uint8` arrays. Grayscale images have shape `(height, width)`; color images have shape `(height, width, 3)` and use RGB channel order after loading.
- `mode` must be exactly `gray` or `color`.
- `k` is the number of segments. Nature-inspired and Otsu methods normally use `k - 1` thresholds per channel. K-means records `k` cluster centroids per channel in the CSV field named `thresholds`.
- A nature-inspired result has a numeric `fitness` and convergence `history`. Classical results set both fields to `None`.
- MEALPY minimizes every registered fitness function. Objectives that conceptually maximize quality therefore return a negated value.
- Each split row has the form `relative-image-path,mode`; blank lines are ignored and missing images are warned about and skipped.

### Extending the project

#### Add a nature-inspired algorithm

1. Implement or import a MEALPY-compatible optimizer class.
2. Import it in `segmentation/base_segmenter.py`.
3. Add its lowercase CLI name and class to `NATURE_ALGO`.
4. Test grayscale and color modes with every intended fitness function.

#### Add a classical algorithm

1. Implement a function with signature `(image, k, mode)` returning `(segmented_image, thresholds)`.
2. Register it in `CLASSICAL_ALGO` in `segmentation/base_segmenter.py`.
3. Confirm that its returned image preserves the input shape and `uint8` type.

#### Add a fitness function

1. Implement a callable with signature `(image, thresholds, k, mode)` returning one finite scalar.
2. Remember that lower is better, or negate a score that should be maximized.
3. Import and register it in `FITNESS_FUNCTIONS` in `segmentation/problem.py`.

#### Add an output metric

1. Implement it in `fitness/reconstruction.py` and return it from `compute_all_metrics()`.
2. Add its column to `CSV_HEADER` and its value to the row dictionary in `evaluation/runner.py`.
3. Validate both grayscale and RGB inputs.

Confirm that the final message reports zero failed tasks and inspect both generated CSV files. Optimizer changes should also be smoke-tested with a reduced epoch/population configuration through `run_segmentation()` before launching the 500-epoch defaults.

## Troubleshooting

### `ModuleNotFoundError`

Confirm that the virtual environment is active and reinstall the requirements:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

### Every task reports `FAILED`

Check that:

- the algorithm and fitness names exactly match the lists above;
- `k` is an integer of at least 2 and is sensible for the image;
- the split exists in `data/splits/`; and
- every split line uses `filename,gray` or `filename,color`.

### Running the project after moving it

No source file contains a machine-specific project path. Data, splits, imports, and results are resolved relative to the repository itself. After moving or cloning the directory, create a new virtual environment there and reinstall `requirements.txt`; existing virtual environments should not be moved between locations.
