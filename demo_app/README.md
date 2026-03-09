# FastAPI Application Guide

This directory contains the backend runtime for the retinal demo application. It is responsible for loading trained model artifacts, checking that the literature index is ready, serving HTTP endpoints, and orchestrating the image-to-answer workflow.

## What This Backend Owns

- [`main.py`](./main.py): FastAPI app factory, startup lifecycle, and HTTP routes
- [`artifacts.py`](./artifacts.py): artifact sync and validation
- [`inference.py`](./inference.py): model loading and image prediction
- [`workflow.py`](./workflow.py): image-summary and follow-up orchestration
- [`config.py`](./config.py): demo-facing config wrapper over the shared settings layer

## Startup Lifecycle

When the app starts, [`main.py`](./main.py) builds a runtime in this order:

1. Sync or validate model artifacts.
2. Build the RAG config.
3. Inspect the Chroma index and fail if it is not ready.
4. Load the trained classifier from `best_model.pt`, `label_map.json`, and `run_config.json`.
5. Construct the initial and follow-up workflows.
6. Expose the runtime on the FastAPI app state.

If any of those steps fail, `/api/health` reports the startup error and the app remains unready.

## Required Model Artifacts

The backend expects these files in the resolved artifact directory:

- `best_model.pt`
- `label_map.json`
- `run_config.json`

`metrics.json` is optional but is loaded when present.

These files are the contract between the training pipeline and the runtime. Artifact validation is implemented in [`artifacts.py`](./artifacts.py).

## How to Run the Backend

Preferred commands:

```bash
make api
make dev
```

- `make api` starts only the FastAPI backend
- `make dev` starts both FastAPI and the Reflex frontend through [`../scripts/run_dev.py`](../scripts/run_dev.py)

Direct Uvicorn command:

```bash
uvicorn demo_app.main:create_app --factory --reload
```

## Configuration

The backend reads settings from [`../app_config.py`](../app_config.py) through [`config.py`](./config.py).

Important variables:

- `MODEL_ARTIFACT_SOURCE_URL`
- `MODEL_ARTIFACT_LOCAL_DIR`
- `MODEL_ARTIFACT_FORCE_REFRESH`
- `MODEL_ARTIFACT_RUN_SUBDIR`
- `CLASSIFIER_DEVICE`
- `CLASSIFIER_LOW_CONF_THRESHOLD`
- `CLASSIFIER_TOP_K`
- `APP_RAG_TOP_K`
- `APP_TITLE`
- `API_HOST`
- `API_PORT`

Google Drive folders and local directory sources are both supported for artifacts. If `MODEL_ARTIFACT_LOCAL_DIR` already contains a valid run, the backend can start without `MODEL_ARTIFACT_SOURCE_URL`. If the local cache is empty, invalid, or `MODEL_ARTIFACT_FORCE_REFRESH=true`, then `MODEL_ARTIFACT_SOURCE_URL` becomes required. If multiple run directories are downloaded, `MODEL_ARTIFACT_RUN_SUBDIR` must be set explicitly.

## API Surface

The backend serves four primary routes:

- `GET /`
- `GET /api/health`
- `POST /api/predict`
- `POST /api/answer`
- `POST /api/followup`

Behavior:

- `/api/predict` runs classifier inference only
- `/api/answer` runs classifier inference plus the literature workflow
- `/api/followup` runs a condition-aware follow-up question against the RAG layer

The main response payloads include prediction data, answer text, citations, warnings, and timings.

## Inference Behavior

[`inference.py`](./inference.py) reconstructs the model from the saved training contract:

- it reads `label_map.json` to recover class ordering
- it reads `run_config.json` to recover the model name and image size
- it loads `best_model.pt` and returns sorted top-k predictions
- it emits a warning when the top prediction falls below the configured confidence threshold

That means any incompatible change to the training artifact format must also update the runtime loader.

## How to Modify the Backend

Common changes and where they belong:

- change startup validation or artifact source rules: [`artifacts.py`](./artifacts.py)
- change model loading or prediction payloads: [`inference.py`](./inference.py)
- add or change HTTP routes: [`main.py`](./main.py)
- adjust question-building or follow-up behavior: [`workflow.py`](./workflow.py)

If you want to modify the LangChain-style orchestration behavior specifically, start with [`../docs/langchain/README.md`](../docs/langchain/README.md).

## Common Failure Modes

- Missing required artifacts: verify the resolved run directory contains the required files.
- RAG not ready: run `make inspect-index` and confirm the Chroma index exists.
- Invalid uploads: `/api/predict` and `/api/answer` reject empty or invalid image files.
- Multiple artifact runs downloaded: set `MODEL_ARTIFACT_RUN_SUBDIR`.

## Where to Read Next

- [`../README.md`](../README.md): repo overview
- [`../docs/langchain/README.md`](../docs/langchain/README.md): workflow orchestration details
