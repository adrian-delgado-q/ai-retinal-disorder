# LlamaIndex + Chroma RAG Guide

This directory owns the literature retrieval subsystem. It indexes open-access ophthalmology papers, stores embeddings in Chroma, and exposes retrieval and answer-generation helpers that the demo application can call directly or through the LangChain workflow wrapper.

## Source of Truth

The RAG source of truth is [`../data/articles/eye_conditions_open_access_database.json`](../data/articles/eye_conditions_open_access_database.json).

At the current repository snapshot, that dataset contains:

- 50 article records
- all records sourced from PMC
- all records marked `high` trust
- 37 downloaded local PDFs
- 13 failed downloads

Each record provides article metadata plus the local PDF path used during indexing.

## What This Subsystem Does

The RAG pipeline is:

`metadata JSON -> downloaded PDFs -> text extraction -> chunking -> embeddings -> Chroma -> retrieval -> answer synthesis`

Concretely:

- [`metadata.py`](./metadata.py): validates and normalizes article records
- [`parsing.py`](./parsing.py): extracts PDF text and trims likely back matter
- [`chunking.py`](./chunking.py): splits documents into section-aware chunks
- [`ids.py`](./ids.py): generates deterministic document and node IDs
- [`index_builder.py`](./index_builder.py): prepares nodes and upserts them into Chroma
- [`retrieval.py`](./retrieval.py): runs filtered retrieval and optional synthesis
- [`agent_tools.py`](./agent_tools.py): exposes structured tool wrappers around retrieval and answer generation

## How to Build and Inspect the Index

Preferred commands:

```bash
make build-index
make inspect-index
```

Direct CLI commands:

```bash
python scripts/index_articles.py build-index
python scripts/index_articles.py inspect-index
```

The build command reads the article JSON, skips missing or non-downloadable records, parses the PDFs that exist, generates embeddings, and upserts the resulting chunks into the configured Chroma collection.

The inspect command reports:

- dataset path
- chunk count
- indexed document count
- source and trust-level distributions
- missing files
- metadata completeness

## How to Query the Index

Raw retrieval:

```bash
python scripts/index_articles.py query-raw "What are common glaucoma risk factors?"
```

Synthesized answer with citations:

```bash
python scripts/index_articles.py query-answer "Summarize common diabetic retinopathy treatments."
```

Useful filters:

- `--disease-tag`
- `--source`
- `--year-min`
- `--year-max`
- `--trust-level`
- `--top-k`
- `--allow-low-trust`

`query-answer` requires `DEEPSEEK_API_KEY` because synthesis is performed by the configured LlamaIndex-compatible LLM wrapper in [`retrieval.py`](./retrieval.py).

## Configuration

This subsystem reads its defaults from the centralized settings layer in [`../app_config.py`](../app_config.py), exposed through [`config.py`](./config.py).

Key variables:

- `ARTICLE_DATASET_PATH`
- `CHROMA_DIR`
- `CHROMA_COLLECTION`
- `EMBED_MODEL_NAME`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_EMBED_BATCH_SIZE`
- `RAG_TOP_K`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`
- `DEEPSEEK_BASE_URL`

Inspect the resolved values with:

```bash
make config-check
```

## Retrieval Behavior

- retrieval excludes low-trust content by default
- explicit `trust_levels` or `--allow-low-trust` can override that baseline
- metadata filters are pushed into Chroma where possible and then re-applied in Python for correctness
- `query_raw()` returns scored chunks and metadata directly
- `synthesize_answer()` first retrieves chunks, then builds a grounded prompt and asks the configured LLM to answer only from that context

## How to Modify This Layer

Common changes and where to make them:

- change metadata contract: [`metadata.py`](./metadata.py)
- change parsing heuristics: [`parsing.py`](./parsing.py)
- change chunk sizes or section strategy: [`chunking.py`](./chunking.py) and config
- change retrieval filters or scoring behavior: [`retrieval.py`](./retrieval.py)
- change tool schema or tool names: [`agent_tools.py`](./agent_tools.py)

If you change chunking or metadata semantics, rebuild the index so the Chroma collection reflects the new contract.

## Common Failure Modes

- Missing PDFs: inspect the `path` fields in the article JSON and rerun `make inspect-index`.
- Empty query results: remove over-constraining filters and inspect metadata distributions.
- Embedding failures: verify the configured Hugging Face embedding model is available.
- Synthesis failures: verify `DEEPSEEK_API_KEY` and the configured API endpoint.

## Where to Read Next

- [`../README.md`](../README.md): repo overview
- [`../docs/langchain/README.md`](../docs/langchain/README.md): how the LangChain workflow wraps this retrieval layer
