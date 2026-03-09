# Retinal ML + RAG Platform

This repository is a retinal image classification and literature-grounded question-answering platform. It combines curated retinal fundus image manifests, Colab-based model training, a FastAPI inference backend, a Reflex frontend, and a LlamaIndex + Chroma retrieval layer built from open-access ophthalmology literature. The end-to-end flow is straightforward: a user uploads a retinal image, the classifier predicts a likely condition, the RAG subsystem retrieves relevant PMC literature, and the application returns a synthesized answer with citations and runtime metadata.

![Demo](/evidence_media/output-2x-ai-rag.gif)


## Why This Project Exists

Eye disease detection matters because delayed intervention can lead to avoidable vision loss. The external Eye Disease Image Dataset cited below frames this clearly: eye conditions are a major source of non-fatal disability, and early detection improves the odds of timely treatment and preserved quality of life. This project uses retinal images plus open-access literature to support research, prototyping, and internal demos around that workflow.

This repository is not a clinical diagnostic system. It is an engineering and research platform for model training, retrieval, and grounded explanation.

## Data Sources

### Eye Image Dataset

The image pipeline is derived from the external **Eye Disease Image Dataset**:

- Published: 2 April 2024
- Version: 1
- DOI: `10.17632/s9bfhswzjb.1`
- Contributors: Mohammad Riadur Rashid, Shayla Sharmin, Tania Khatun, Md Zahid Hasan, Mohammad Shorif Uddin

Inside this repo, the image data is organized as a local curation pipeline rather than a single raw drop:

- `data/raw/` stores source retinal images.
- `data/reviewed/labels_master.csv` is the row-level manifest used to track labels, split assignment, review status, image quality, and modality checks.
- `data/reviewed/dataset_summary.csv` is the class and split summary used to understand coverage.
- `data/splits/` contains the train, validation, and test CSV manifests consumed by the training code.

The broader reviewed and split pipeline currently contains **9 included classes**:

- central serous chorioretinopathy
- diabetic retinopathy
- disc edema
- glaucoma
- healthy
- macular scar
- myopia
- retinal detachment
- retinitis pigmentosa

`pterygium` appears in the reviewed summary but is currently excluded from the curated split set.

The active training code narrows that broader curated dataset to **5 target classes** in [`training/config.py`](training/config.py):

- diabetic retinopathy
- glaucoma
- healthy
- myopia
- macular scar

That distinction matters: the repository contains a wider reviewed image dataset than the current production demo model actually serves.

### Open-Access Literature Dataset

The source of truth for the RAG subsystem is [`data/articles/eye_conditions_open_access_database.json`](data/articles/eye_conditions_open_access_database.json).

The current repository snapshot contains:

- 50 article records
- all records sourced from PMC
- all records marked `high` trust
- 37 records downloaded locally
- 13 records marked as failed downloads
- an even 10-record distribution across 5 condition tags:
  - diabetic retinopathy
  - glaucoma
  - healthy
  - macular scar
  - myopia

These records supply the metadata and local PDF paths used by the LlamaIndex ingestion pipeline, which parses PDFs, chunks text, and stores embeddings in Chroma for retrieval.

## How the Full Application Works

![AI Eye Evaluation Diagram](/evidence_media/AI-Eye-Evaluation-Diagram.jpeg)

The full workflow looks like this:

1. Retinal images are curated into reviewed manifests and split CSVs.
2. A classifier is trained in Colab using the code in `training/` and notebooks in `colab/`.
3. Training exports model artifacts such as `best_model.pt`, `label_map.json`, and `run_config.json`.
4. The RAG pipeline reads [`data/articles/eye_conditions_open_access_database.json`](data/articles/eye_conditions_open_access_database.json), parses downloaded PDFs, and builds a Chroma index.
5. The demo app starts and loads the trained vision artifacts plus the indexed literature store.
6. A user uploads a retinal image through the frontend or API.
7. The backend runs model inference and returns top predictions and confidence scores.
8. The predicted label helps shape the literature query.
9. LlamaIndex + Chroma retrieve relevant article chunks.
10. The answer layer synthesizes a grounded response with citations, warnings, and timings.

Compact pipeline:

`retinal image -> classifier prediction -> condition-aware RAG query -> PMC literature retrieval -> AI synthesis with citations`

## What Each Technology Contributes

- **ML**: `timm`, `torch`, and manifest-based dataset loading power the retinal image classifier.
- **Colab**: notebooks in `colab/` support training and validation workflows for the vision model.
- **AI / LLM**: the answer layer synthesizes retrieved evidence into a readable response.
- **RAG**: LlamaIndex, Chroma, and PDF parsing provide grounded retrieval over open-access ophthalmology papers.
- **Existing datasets**: the retinal image corpus supports classification, while the PMC article corpus supports evidence-backed explanations.

## Repository Structure

```text
.
├── app_config.py
├── colab/
├── data/
│   ├── articles/
│   ├── model_artifacts/
│   ├── raw/
│   ├── reviewed/
│   └── splits/
├── demo_app/
├── docs/
├── rag/
├── reflex_frontend/
├── scripts/
└── training/
```

- `app_config.py`: centralized typed settings for training, RAG, backend, and frontend.
- `colab/`: notebook-based training and validation flow for the retinal classifier.
- `data/raw/`: source retinal images used to build the curated manifests.
- `data/reviewed/`: reviewed label manifests and dataset summaries.
- `data/splits/`: train, validation, and test CSVs consumed by the training pipeline.
- `data/articles/`: literature metadata JSON, downloaded PDFs, and Chroma index storage.
- `data/model_artifacts/`: local cache for trained model artifacts used by the demo app.
- `training/`: dataset loading, transforms, training loop, and evaluation utilities.
- `rag/`: article normalization, PDF parsing, chunking, indexing, retrieval, and agent-facing tools.
- `demo_app/`: artifact sync, inference engine, image-to-answer workflows, and FastAPI API.
- `reflex_frontend/`: upload and question UI for interacting with the backend.
- `scripts/`: operational utilities such as indexing, config inspection, and local dev launch.
- `docs/`: deeper subsystem documentation and implementation notes.

## Component Guide

- `training/`: loads manifest-driven datasets, applies transforms, trains the classifier, and saves evaluation artifacts.
- `rag/`: turns article metadata and PDFs into searchable chunks, applies retrieval filters, and exposes answer-generation tools.
- `demo_app/`: downloads or validates model artifacts, loads the classifier, runs image-to-answer workflows, and serves the API.
- `reflex_frontend/`: provides the user-facing interface for uploads, predictions, answers, and follow-up questions.
- `app_config.py`: defines the shared settings layer used across training, indexing, backend, and frontend runtimes.

## Running the System

For the fastest happy path:

```bash
make install
cp .env.example .env
make config-check
make build-index
make dev
```

Use [`demo_app/README.md`](demo_app/README.md) for backend, frontend, and API details. Use [`rag/README.md`](rag/README.md) and [`colab/README.md`](colab/README.md) for subsystem-specific execution guides.

## System Behavior and Outputs

The application exposes three main runtime surfaces:

- prediction-only: classify an uploaded retinal image and return label probabilities
- image-to-answer: classify the image, run literature retrieval, and return a synthesized response
- follow-up question: ask a condition-aware follow-up question against the literature index

The end-to-end app returns:

- prediction labels and confidence
- top prediction list
- answer text
- citations
- warnings
- timings

For concrete endpoint behavior, see [`demo_app/README.md`](demo_app/README.md).

## Safety and Limitations

- This system is not for clinical diagnosis or treatment decisions.
- The literature layer is retrieval-grounded, but synthesis can still be incomplete or wrong.
- The current RAG corpus is limited in size and only includes the downloaded subset of the tracked article list.
- The active model scope is narrower than the full reviewed retinal dataset because training currently targets 5 classes.
- Any real-world use would require human review, medical oversight, and a stronger validation process than this repository currently provides.

## Further Reading

- [`colab/README.md`](colab/README.md): how to run and validate Colab-based training
- [`rag/README.md`](rag/README.md): how LlamaIndex + Chroma indexing and retrieval work
- [`docs/langchain/README.md`](docs/langchain/README.md): how the LangChain workflow executes and how to modify it
- [`demo_app/README.md`](demo_app/README.md): how the FastAPI backend loads artifacts and serves the app
