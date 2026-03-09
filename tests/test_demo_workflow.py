from __future__ import annotations

import pytest

from demo_app.workflow import (
    DEFAULT_INITIAL_SUMMARY_TEMPLATES,
    FollowupQuestionWorkflow,
    ImageSummaryWorkflow,
    build_followup_rag_query,
    build_initial_rag_query,
)
from rag.config import RAGConfig


class FakeInferenceEngine:
    class_names = ["diabetic_retinopathy", "glaucoma", "healthy", "myopia", "macular_scar"]

    def __init__(self, label: str, warning: str | None = None):
        self.label = label
        self.warning = warning

    def predict_bytes(self, _image_bytes, *, top_k=3):
        return type(
            "Prediction",
            (),
            {
                "predicted_label": self.label,
                "predicted_confidence": 0.82,
                "top_predictions": [],
                "warning": self.warning,
                "to_dict": lambda self_: {
                    "label": self_.predicted_label,
                    "confidence": self_.predicted_confidence,
                    "top_predictions": [],
                    "warning": self_.warning,
                },
            },
        )()


def test_build_initial_rag_query_uses_fixed_prompt_and_nonhealthy_prediction():
    query = build_initial_rag_query(
        predicted_label="glaucoma",
        top_k=2,
    )
    assert query.question == DEFAULT_INITIAL_SUMMARY_TEMPLATES["glaucoma"]
    assert query.disease_tag == "glaucoma"


def test_build_initial_rag_query_keeps_healthy_unfiltered():
    query = build_initial_rag_query(
        predicted_label="healthy",
        top_k=2,
    )
    assert query.disease_tag is None


def test_initial_prompt_templates_cover_all_supported_classes():
    expected = {"diabetic_retinopathy", "glaucoma", "healthy", "myopia", "macular_scar"}
    assert set(DEFAULT_INITIAL_SUMMARY_TEMPLATES) == expected


def test_build_followup_rag_query_uses_condition_and_question():
    query = build_followup_rag_query(
        condition="glaucoma",
        question="What are common treatments?",
        top_k=2,
        known_labels=["glaucoma", "healthy"],
    )
    assert query.question == "What are common treatments?"
    assert query.disease_tag == "glaucoma"


def test_build_followup_rag_query_keeps_healthy_unfiltered_without_match():
    query = build_followup_rag_query(
        condition="healthy",
        question="What should a normal retinal screen include?",
        top_k=2,
        known_labels=["glaucoma", "healthy"],
    )
    assert query.disease_tag is None


def test_build_followup_rag_query_requires_question():
    with pytest.raises(ValueError):
        build_followup_rag_query(
            condition="glaucoma",
            question="   ",
            top_k=2,
            known_labels=["glaucoma", "healthy"],
        )


def test_initial_workflow_returns_prediction_summary_and_warnings():
    captured = {}

    def fake_answer(payload, *, config=None):
        captured["payload"] = payload
        captured["config"] = config
        return {"answer": "Grounded answer.", "citations": [{"text": "chunk"}]}

    workflow = ImageSummaryWorkflow(
        inference_engine=FakeInferenceEngine("healthy", warning="low confidence"),
        rag_config=RAGConfig.from_env(),
        default_top_k=3,
        answer_fn=fake_answer,
    )

    result = workflow.run(image_bytes=b"fake", top_k=2)

    assert result["prediction"]["label"] == "healthy"
    assert result["rag_query"]["disease_tag"] is None
    assert result["initial_summary"]["text"] == "Grounded answer."
    assert result["warnings"] == ["low confidence"]
    assert captured["payload"].question == DEFAULT_INITIAL_SUMMARY_TEMPLATES["healthy"]


def test_followup_workflow_returns_answer_for_condition_and_question():
    captured = {}

    def fake_answer(payload, *, config=None):
        captured["payload"] = payload
        return {"answer": "Follow-up answer.", "citations": [{"text": "chunk"}]}

    workflow = FollowupQuestionWorkflow(
        known_labels=["diabetic_retinopathy", "glaucoma", "healthy", "myopia", "macular_scar"],
        rag_config=RAGConfig.from_env(),
        default_top_k=3,
        answer_fn=fake_answer,
    )

    result = workflow.run(condition="glaucoma", question="What treatments are common?", top_k=2)

    assert result["condition"] == "glaucoma"
    assert result["answer"]["text"] == "Follow-up answer."
    assert captured["payload"].question == "What treatments are common?"
    assert captured["payload"].disease_tag == "glaucoma"
