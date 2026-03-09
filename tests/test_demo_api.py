from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from demo_app.config import DemoAppConfig
from demo_app.main import DemoRuntime, create_app


class FakeInferenceEngine:
    def predict_bytes(self, _image_bytes, *, top_k=3):
        return type(
            "Prediction",
            (),
            {
                "to_dict": lambda self_: {
                    "label": "glaucoma",
                    "confidence": 0.91,
                    "top_predictions": [{"label": "glaucoma", "probability": 0.91}],
                    "warning": None,
                },
                "warning": None,
            },
        )()


class FakeWorkflow:
    def run(self, *, image_bytes, top_k=None, prediction_top_k=3):
        return {
            "prediction": {
                "label": "glaucoma",
                "confidence": 0.91,
                "top_predictions": [{"label": "glaucoma", "probability": 0.91}],
                "warning": None,
            },
            "initial_summary": {"text": "Common symptoms and treatments."},
            "rag_query": {
                "question": "Summarize common symptoms and common treatments for glaucoma.",
                "disease_tag": "glaucoma",
                "top_k": top_k,
            },
            "citations": [{"text": "chunk"}],
            "timings": {"classification_ms": 1.0, "rag_ms": 2.0},
            "warnings": [],
        }


class FakeFollowupWorkflow:
    def run(self, *, condition, question, top_k=None):
        return {
            "condition": condition,
            "question": question,
            "answer": {"text": "Follow-up answer."},
            "citations": [{"text": "chunk"}],
            "timings": {"rag_ms": 2.0},
            "warnings": [],
        }


def _build_app():
    runtime = DemoRuntime(
        artifact_bundle=type("Artifacts", (), {"artifact_dir": "tmp"})(),
        inference_engine=FakeInferenceEngine(),
        initial_workflow=FakeWorkflow(),
        followup_workflow=FakeFollowupWorkflow(),
        rag_status={"index_ready": True, "chunk_count": 10},
    )
    config = DemoAppConfig(model_artifact_source_url="unused")
    return create_app(config, runtime=runtime)


async def _request(app, method: str, path: str, **kwargs):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def test_health_endpoint_reports_ready_state():
    cold_app = create_app(DemoAppConfig(model_artifact_source_url="unused"))
    assert cold_app.state.ready is False

    response = asyncio.run(_request(_build_app(), "GET", "/api/health"))
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_predict_endpoint_returns_prediction_payload():
    response = asyncio.run(
        _request(
            _build_app(),
            "POST",
            "/api/predict",
            files={"image": ("retina.png", b"fake-image", "image/png")},
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["label"] == "glaucoma"


def test_answer_endpoint_returns_combined_payload():
    response = asyncio.run(
        _request(
            _build_app(),
            "POST",
            "/api/answer",
            files={"image": ("retina.png", b"fake-image", "image/png")},
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["label"] == "glaucoma"
    assert body["initial_summary"]["text"] == "Common symptoms and treatments."


def test_followup_endpoint_returns_answer_payload():
    response = asyncio.run(
        _request(
            _build_app(),
            "POST",
            "/api/followup",
            json={"condition": "glaucoma", "question": "What treatments are common?"},
        )
    )
    assert response.status_code == 200
    body = response.json()
    assert body["condition"] == "glaucoma"
    assert body["answer"]["text"] == "Follow-up answer."


def test_followup_endpoint_validates_required_fields():
    response = asyncio.run(
        _request(
            _build_app(),
            "POST",
            "/api/followup",
            json={"condition": "", "question": ""},
        )
    )
    assert response.status_code == 400


def test_app_returns_json_for_unhandled_errors():
    class CrashingWorkflow:
        def run(self, *, image_bytes, top_k=None, prediction_top_k=3):
            raise RuntimeError("boom")

    runtime = DemoRuntime(
        artifact_bundle=type("Artifacts", (), {"artifact_dir": "tmp"})(),
        inference_engine=FakeInferenceEngine(),
        initial_workflow=CrashingWorkflow(),
        followup_workflow=FakeFollowupWorkflow(),
        rag_status={"index_ready": True, "chunk_count": 10},
    )
    app = create_app(DemoAppConfig(model_artifact_source_url="unused"), runtime=runtime)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/api/answer",
            files={"image": ("retina.png", b"fake-image", "image/png")},
        )
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error. Check backend logs for the stack trace."
