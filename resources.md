# Large Resources

The following directories are excluded from version control because they contain large binary assets. This document explains what each directory holds and how to populate it using the project's existing infrastructure.

---

## `data/articles/downloads/`

**Contents:** PDF files downloaded from PubMed Central (PMC) for the 50 articles tracked in `data/articles/eye_conditions_open_access_database.json`.

**How to fetch:**

```bash
make download-articles
```

Or with additional arguments (e.g. to re-download everything):

```bash
make download-articles ARGS="--overwrite"
```

The script contacts the PMC Open Access API to resolve each article's PDF. A `manifest.json` is written to `data/articles/downloads/` recording the result for every article.

**Expected outcome:** 37 of 50 PDFs will download successfully. The remaining 13 are from the PMC non-OA subset and will have `"status": "failed"` in the manifest — this is expected behaviour and does not require any action.

**Next step:** Once the PDFs are present, build the Chroma vector index:

```bash
make build-index
```

This must be done before starting the demo app with RAG functionality.

---

## `data/model_artifacts/`

**Contents:** Trained `efficientnet_b0` model weights and metadata: `best_model.pt`, `last_model.pt`, `label_map.json`, `run_config.json`, `metrics.json`, `history.csv`.

There are two ways to populate this directory.

### Option A — Pull from a remote source (recommended)

Set `MODEL_ARTIFACT_SOURCE_URL` in your `.env` file to a supported location. The demo app will fetch the artifacts automatically on startup.

```dotenv
# .env
MODEL_ARTIFACT_SOURCE_URL=https://drive.google.com/drive/folders/<folder-id>
```

Supported URL types:
- A public **Google Drive folder** URL (requires `gdown`, included in `requirements.txt`)
- A **local filesystem path** (absolute or relative)
- A `file://` URI

On startup the app will download and cache the artifacts to `data/model_artifacts/`. Subsequent starts reuse the cached copy unless `force_refresh` is triggered.

### Option B — Retrain locally

> Requires `data/raw/` to be populated first (see below).

```bash
make train
```

Artifacts are written to `runs/default/` (or the path configured in `TRAIN_OUTPUT_DIR`). Copy them to the expected location:

```bash
cp -r runs/default/* data/model_artifacts/
```

The demo app requires these three files to start:

| File | Description |
|---|---|
| `best_model.pt` | Best-epoch model weights |
| `label_map.json` | Class index → label name mapping |
| `run_config.json` | Architecture and pre-processing metadata |

`metrics.json` and `history.csv` are optional and used only for display purposes.

---

## `data/raw/`

**Contents:** Retinal fundus images organised by class, used for training and evaluation.

Expected structure:

```
data/raw/
└── original/
    ├── Central_Serous_Chorioretinopathy/
    ├── Diabetic_Retinopathy/
    ├── Disc_Edema/
    ├── Glaucoma/
    ├── Healthy/
    ├── Macular_Scar/
    ├── Myopia/
    ├── Pterygium/
    ├── Retinal_Detachment/
    └── Retinitis_Pigmentosa/
```

**How to fetch:** There is currently no automated download script for this dataset. The images were originally sourced from a shared Google Drive used during Colab training.

1. Obtain access to the shared Google Drive folder containing the raw retinal images.
2. Download the folder contents and place them under `data/raw/original/`, preserving the per-class subdirectory structure shown above.

The training pipeline reads from these class folders via the split CSVs in `data/splits/`. Note that training currently uses only 5 of the 10 classes (`diabetic_retinopathy`, `glaucoma`, `healthy`, `macular_scar`, `myopia`); the remaining class folders are kept for completeness and potential future use.
