from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from html import escape

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse

from rag.index_builder import inspect_index

from .artifacts import ArtifactBundle, sync_model_artifacts
from .config import DemoAppConfig
from .inference import RetinalInferenceEngine
from .workflow import FollowupQuestionWorkflow, ImageSummaryWorkflow
from runtime_logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoRuntime:
    artifact_bundle: ArtifactBundle
    inference_engine: RetinalInferenceEngine
    initial_workflow: ImageSummaryWorkflow
    followup_workflow: FollowupQuestionWorkflow
    rag_status: dict


def initialize_runtime(config: DemoAppConfig) -> DemoRuntime:
    logger.info(
        "Initializing demo runtime with artifact_dir=%s run_subdir=%s device=%s",
        config.model_artifact_local_dir,
        config.model_artifact_run_subdir,
        config.classifier_device,
    )
    artifact_bundle = sync_model_artifacts(
        source_url=config.model_artifact_source_url,
        local_dir=config.model_artifact_local_dir,
        force_refresh=config.model_artifact_force_refresh,
        run_subdir=config.model_artifact_run_subdir,
    )
    rag_config = config.build_rag_config()
    rag_status = inspect_index(rag_config)
    if not rag_status.get("index_ready"):
        raise RuntimeError(f"RAG index is not ready: {rag_status}")
    inference_engine = RetinalInferenceEngine.from_artifacts(
        artifact_bundle.artifact_dir,
        device=config.classifier_device,
        low_confidence_threshold=config.classifier_low_conf_threshold,
    )
    initial_workflow = ImageSummaryWorkflow(
        inference_engine=inference_engine,
        rag_config=rag_config,
        default_top_k=config.rag_top_k,
    )
    followup_workflow = FollowupQuestionWorkflow(
        known_labels=inference_engine.class_names,
        rag_config=rag_config,
        default_top_k=config.rag_top_k,
    )
    return DemoRuntime(
        artifact_bundle=artifact_bundle,
        inference_engine=inference_engine,
        initial_workflow=initial_workflow,
        followup_workflow=followup_workflow,
        rag_status=rag_status,
    )


def create_app(
    config: DemoAppConfig | None = None,
    *,
    runtime: DemoRuntime | None = None,
) -> FastAPI:
    config = config or DemoAppConfig.from_env()

    if runtime is None:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.ready = False
            app.state.startup_error = None
            try:
                app.state.runtime = initialize_runtime(config)
                app.state.ready = True
                yield
            except Exception as exc:
                app.state.startup_error = str(exc)
                raise

        app = FastAPI(title=config.app_title, lifespan=lifespan)
        app.state.ready = False
        app.state.startup_error = None
        app.state.runtime = None
    else:
        app = FastAPI(title=config.app_title)
        app.state.ready = True
        app.state.startup_error = None
        app.state.runtime = runtime

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request error for %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error. Check backend logs for the stack trace."},
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "%s %s -> %s in %sms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        return HTMLResponse(_render_home_page(config.app_title))

    @app.get("/api/health")
    async def health() -> dict:
        runtime_state = getattr(app.state, "runtime", None)
        rag_status = runtime_state.rag_status if runtime_state is not None else None
        artifact_dir = None
        if runtime_state is not None:
            artifact_dir = str(runtime_state.artifact_bundle.artifact_dir)
        return {
            "ready": bool(getattr(app.state, "ready", False)),
            "startup_error": getattr(app.state, "startup_error", None),
            "artifact_dir": artifact_dir,
            "index_ready": rag_status.get("index_ready") if rag_status else False,
            "chunk_count": rag_status.get("chunk_count") if rag_status else None,
        }

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...)) -> dict:
        runtime_state = _require_runtime(app)
        image_bytes = await _read_upload(image)
        logger.info("Received predict request for filename=%s size=%s", image.filename, len(image_bytes))
        try:
            prediction = runtime_state.inference_engine.predict_bytes(
                image_bytes,
                top_k=config.classifier_top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "prediction": prediction.to_dict(),
            "timings": {},
            "warnings": [prediction.warning] if prediction.warning else [],
        }

    @app.post("/api/answer")
    async def answer(
        image: UploadFile = File(...),
        top_k: int | None = Form(default=None),
    ) -> dict:
        runtime_state = _require_runtime(app)
        image_bytes = await _read_upload(image)
        logger.info(
            "Received answer request for filename=%s size=%s top_k=%s",
            image.filename,
            len(image_bytes),
            top_k or config.rag_top_k,
        )
        try:
            result = runtime_state.initial_workflow.run(
                image_bytes=image_bytes,
                top_k=top_k or config.rag_top_k,
                prediction_top_k=config.classifier_top_k,
            )
            logger.info(
                "Answer request completed with label=%s citations=%s warnings=%s",
                result.get("prediction", {}).get("label"),
                len(result.get("citations", [])),
                len(result.get("warnings", [])),
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/followup")
    async def followup(request: Request) -> dict:
        runtime_state = _require_runtime(app)
        payload = await _read_followup_payload(request)
        logger.info(
            "Received followup request for condition=%s top_k=%s",
            payload["condition"],
            payload.get("top_k") or config.rag_top_k,
        )
        try:
            result = runtime_state.followup_workflow.run(
                condition=payload["condition"],
                question=payload["question"],
                top_k=payload.get("top_k") or config.rag_top_k,
            )
            logger.info(
                "Followup completed for condition=%s citations=%s warnings=%s",
                result.get("condition"),
                len(result.get("citations", [])),
                len(result.get("warnings", [])),
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _require_runtime(app: FastAPI) -> DemoRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None or not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Application is not ready.")
    return runtime


async def _read_upload(upload: UploadFile) -> bytes:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="An image file is required.")
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return payload


async def _read_followup_payload(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Follow-up payload must be a JSON object.")
        data = payload
    else:
        form = await request.form()
        data = dict(form)

    condition = str(data.get("condition", "")).strip()
    question = str(data.get("question", "")).strip()
    raw_top_k = data.get("top_k")
    top_k = None
    if raw_top_k not in (None, ""):
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="top_k must be an integer.") from exc

    if not condition:
        raise HTTPException(status_code=400, detail="condition is required.")
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")
    return {"condition": condition, "question": question, "top_k": top_k}


def _render_home_page(title: str) -> str:
    safe_title = escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{safe_title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      color: #1e2a28;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 30%),
        radial-gradient(circle at bottom right, rgba(194, 65, 12, 0.14), transparent 28%),
        #f2ede3;
    }}
    main {{
      max-width: 860px;
      margin: 0 auto;
      padding: 48px 20px 64px;
    }}
    .panel {{
      background: rgba(255, 250, 242, 0.94);
      border: 1px solid rgba(30, 42, 40, 0.12);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 18px 60px rgba(30, 42, 40, 0.08);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(2.4rem, 6vw, 4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    p {{
      color: #66756f;
      line-height: 1.7;
      font-size: 1rem;
    }}
    a {{
      color: #0f766e;
      font-weight: 700;
      text-decoration: none;
    }}
    code {{
      background: rgba(30, 42, 40, 0.06);
      padding: 0.15rem 0.35rem;
      border-radius: 8px;
    }}
    ul {{
      line-height: 1.8;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>{safe_title} API</h1>
      <p>This FastAPI process now serves the backend API only. The interactive frontend has moved to a separate Reflex app for better maintainability.</p>
      <ul>
        <li><code>GET /api/health</code></li>
        <li><code>POST /api/predict</code></li>
        <li><code>POST /api/answer</code></li>
        <li><code>POST /api/followup</code></li>
      </ul>
      <p>Run the Reflex frontend separately with <code>reflex run</code> and point it at this API with <code>DEMO_API_URL</code>.</p>
    </section>
  </main>
</body>
</html>"""


app = create_app()
