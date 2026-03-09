# Colab Training Guide

This directory documents the Colab-facing workflow for training and validating the retinal image classifier. The notebooks here are thin orchestration layers around the Python training code in [`training/`](../training/), so the source of truth for training behavior remains the repo code, not notebook-only logic.

## What Lives Here

- [`train_baseline.ipynb`](./train_baseline.ipynb): notebook workflow for launching a baseline training run
- [`validate_baseline.ipynb`](./validate_baseline.ipynb): notebook workflow for checking a saved run and confirming artifacts can be loaded for inference

## What the Colab Workflow Produces

The training pipeline writes the artifacts that the demo backend later consumes:

- `best_model.pt`
- `last_model.pt`
- `label_map.json`
- `run_config.json`
- `metrics.json`
- `history.csv`
- confusion-matrix images

The most important files for downstream inference are:

- `best_model.pt`
- `label_map.json`
- `run_config.json`

Those are the files expected by [`demo_app/inference.py`](../demo_app/inference.py) and [`demo_app/artifacts.py`](../demo_app/artifacts.py).

## Training Entry Point

The actual training command is the Python module in [`training/train_colab.py`](../training/train_colab.py). The notebook ultimately runs the same interface you can run locally:

```bash
python -m training.train_colab \
  --output-dir runs/baseline_efficientnet_b0 \
  --model efficientnet_b0 \
  --epochs 15
```

You can see the full CLI surface with:

```bash
python -m training.train_colab --help
```

## Dataset and Class Scope

Training is manifest-driven. The loader reads split CSVs from `data/splits/` and filters them through the dataset utilities in [`training/dataset.py`](../training/dataset.py).

Important current scope:

- the broader curated dataset contains 9 included classes
- the active training target is narrowed to 5 classes in [`training/config.py`](../training/config.py):
  - diabetic retinopathy
  - glaucoma
  - healthy
  - myopia
  - macular scar

The manifest resolver defaults to clean reviewed CSVs when available and only falls back to provisional manifests if explicitly allowed.

![Eye-Classifier](/evidence_media/output.png)


## Recommended Colab Flow

1. Open [`train_baseline.ipynb`](./train_baseline.ipynb) in Colab.
2. Mount Drive or otherwise make the repo and data paths accessible to the runtime.
3. Install the repo dependencies required by the training code.
4. Launch the training module from the notebook.
5. Save the run output to a persistent location such as Drive.
6. Open [`validate_baseline.ipynb`](./validate_baseline.ipynb) and point it at the saved run directory.
7. Confirm that `best_model.pt`, `label_map.json`, and `run_config.json` all load successfully.

## Key Runtime Behavior

- image augmentation and evaluation transforms are defined in [`training/train_colab.py`](../training/train_colab.py)
- class weighting, early stopping, checkpointing, and metrics export are handled in [`training/utils.py`](../training/utils.py)
- `run_config.json` is written from the resolved config plus CLI overrides, which makes the downstream inference loader deterministic
- if the training run uses provisional manifests, the output directory name is automatically suffixed with `_provisional`

## Useful Local Commands

If you want to reproduce the same workflow outside Colab:

```bash
make install
cp .env.example .env
make config-check
make train ARGS="--output-dir runs/efficientnet_b0 --epochs 10"
```

## Common Failure Modes

- Missing manifests: verify the expected CSVs exist under `data/splits/`.
- Empty filtered dataset: check review status, modality, and image-quality fields in the manifests.
- Broken image paths: inspect `file_path` values and `TRAIN_DATA_ROOT`.
- Missing downstream artifacts: verify the run wrote `best_model.pt`, `label_map.json`, and `run_config.json`.

## Where to Read Next

- [`../README.md`](../README.md): repo overview and architecture
- [`../training/`](../training/): actual training implementation
- [`../demo_app/README.md`](../demo_app/README.md): how model artifacts are consumed by the FastAPI runtime
