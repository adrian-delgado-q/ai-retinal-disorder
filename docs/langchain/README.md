# LangChain Workflow Guide

This document describes the LangChain-facing orchestration layer that turns a classifier prediction into a literature-grounded answer. The current implementation is intentionally light: LangChain is used as the workflow wrapper around the existing retrieval tools rather than as a separate agent framework that owns the whole application.

## Where the Workflow Lives

The key implementation is in:

- [`../../demo_app/workflow.py`](../../demo_app/workflow.py)
- [`../../rag/agent_tools.py`](../../rag/agent_tools.py)

The workflow layer sits between the FastAPI app and the RAG subsystem:

`FastAPI request -> classifier prediction -> workflow query builder -> RAG tool call -> answer with citations`

## What the Workflow Does

There are two runtime flows:

- **initial image summary** via `ImageSummaryWorkflow`
- **follow-up questions** via `FollowupQuestionWorkflow`

For the initial flow:

1. run classifier inference on the uploaded image
2. build a condition-aware question from the predicted label
3. construct a `MedicalQueryInput`
4. call `answer_with_citations`
5. return prediction data, answer text, citations, timings, and warnings

For the follow-up flow:

1. accept `condition` and `question`
2. normalize the label
3. decide whether to keep or drop the disease filter
4. call the same retrieval/synthesis layer
5. return answer text, citations, timings, and warnings

## Current LangChain Usage

The workflow uses `langchain_core.runnables.RunnableLambda` when LangChain Core is available. If it is unavailable, the code falls back to a local sequential runner so the application can still function.

That logic is implemented in `_build_chain()` in [`../../demo_app/workflow.py`](../../demo_app/workflow.py).

This means the current design is:

- workflow-oriented, not agent-heavy
- deterministic and easy to trace
- thin over the existing RAG tools

## Tool Contracts

The workflow calls the structured interfaces in [`../../rag/agent_tools.py`](../../rag/agent_tools.py):

- `retrieve_medical_chunks`
- `answer_with_citations`

Shared input schema:

- `question`
- `disease_tag`
- `trust_levels`
- `source`
- `year_min`
- `year_max`
- `top_k`
- `allow_low_trust`

In the current demo application, the workflow calls `answer_with_citations` directly, but the same schema and tool wrappers can be reused inside a fuller LangChain agent later.

## Default Prompting Behavior

The initial answer flow uses fixed label-specific templates defined in `DEFAULT_INITIAL_SUMMARY_TEMPLATES` in [`../../demo_app/workflow.py`](../../demo_app/workflow.py).

Current behavior:

- non-healthy predictions set `disease_tag` to the normalized predicted label
- `healthy` predictions do not force a disease filter
- follow-up questions for a healthy condition only reintroduce a disease filter if the question explicitly mentions a known disease label

This keeps the workflow grounded without over-filtering healthy cases.

## How to Modify the Workflow

Common changes and where to make them:

- change the default prompt templates: `DEFAULT_INITIAL_SUMMARY_TEMPLATES`
- change how labels are normalized: `_normalize_label()`
- change healthy vs disease-tag routing: `build_initial_rag_query()` and `build_followup_rag_query()`
- add a new processing step to the initial chain: `ImageSummaryWorkflow.__init__()` and `_build_chain()`
- change the retrieval backend call: `_answer_step()` or the `answer_fn` dependency injection

Useful extension patterns:

- add a post-processing step that rewrites or formats citations
- add a moderation or safety-check step before synthesis
- swap `answer_with_citations` for a raw retrieval first-pass plus a custom synthesis function
- route different questions to different prompt templates based on condition or confidence

## How to Execute This Layer

There is no standalone CLI for the workflow layer today. It runs through the application:

```bash
make api
make frontend
```

or:

```bash
make dev
```

Then use:

- `POST /api/answer` for the initial image-to-answer flow
- `POST /api/followup` for the follow-up flow

## Failure Behavior

The workflow is designed to degrade gracefully:

- if the classifier emits a low-confidence result, the warning is surfaced to the client
- if literature synthesis fails, the workflow returns a fallback answer plus a warning instead of crashing the request
- if LangChain Core is unavailable, the fallback sequential chain still executes

That behavior is implemented directly in [`../../demo_app/workflow.py`](../../demo_app/workflow.py).

## Where to Read Next

- [`../../README.md`](../../README.md): repo overview
- [`../../rag/README.md`](../../rag/README.md): indexing and retrieval layer
- [`../../demo_app/README.md`](../../demo_app/README.md): FastAPI runtime and endpoints
