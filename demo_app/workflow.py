from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from rag._deps import require_module
from rag.agent_tools import MedicalQueryInput, answer_with_citations
from rag.config import RAGConfig

from .inference import PredictionResult, RetinalInferenceEngine

logger = logging.getLogger(__name__)


DEFAULT_INITIAL_SUMMARY_TEMPLATES = {
    "diabetic_retinopathy": (
        "Summarize common symptoms and common treatments for diabetic retinopathy."
    ),
    "glaucoma": "Summarize common symptoms and common treatments for glaucoma.",
    "healthy": (
        "Summarize what healthy retinal findings generally imply and when follow-up evaluation may still be appropriate."
    ),
    "myopia": "Summarize common symptoms and common treatments for myopia.",
    "macular_scar": "Summarize common symptoms and common treatments for macular scar.",
}


@dataclass(frozen=True)
class BuiltRagQuery:
    question: str
    disease_tag: str | None
    top_k: int


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _question_mentions_label(question: str, label: str) -> bool:
    normalized_question = _normalize_label(question)
    normalized_label = _normalize_label(label)
    return normalized_label in normalized_question


def build_initial_rag_query(
    *,
    predicted_label: str,
    top_k: int,
) -> BuiltRagQuery:
    predicted_normalized = _normalize_label(predicted_label)
    return BuiltRagQuery(
        question=DEFAULT_INITIAL_SUMMARY_TEMPLATES.get(
            predicted_normalized,
            f"Summarize common symptoms and common treatments for {predicted_label}.",
        ),
        disease_tag=None if predicted_normalized == "healthy" else predicted_normalized,
        top_k=top_k,
    )


def build_followup_rag_query(
    *,
    condition: str,
    question: str,
    top_k: int,
    known_labels: list[str],
) -> BuiltRagQuery:
    normalized_condition = _normalize_label(condition)
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("A follow-up question is required.")

    known_normalized = {_normalize_label(label) for label in known_labels}
    if normalized_condition == "healthy":
        disease_tag = None
        for label in known_labels:
            normalized_label = _normalize_label(label)
            if normalized_label != "healthy" and _question_mentions_label(cleaned_question, label):
                disease_tag = normalized_label
                break
    elif normalized_condition in known_normalized:
        disease_tag = normalized_condition
    else:
        disease_tag = None

    return BuiltRagQuery(question=cleaned_question, disease_tag=disease_tag, top_k=top_k)


class ImageSummaryWorkflow:
    def __init__(
        self,
        *,
        inference_engine: RetinalInferenceEngine,
        rag_config: RAGConfig,
        default_top_k: int,
        answer_fn: Callable[..., dict] = answer_with_citations,
    ) -> None:
        self.inference_engine = inference_engine
        self.rag_config = rag_config
        self.default_top_k = default_top_k
        self.answer_fn = answer_fn
        self._chain = _build_chain(
            self._predict_step,
            self._build_query_step,
            self._answer_step,
        )

    def run(
        self,
        *,
        image_bytes: bytes,
        top_k: int | None = None,
        prediction_top_k: int = 3,
    ) -> dict:
        state = {
            "image_bytes": image_bytes,
            "top_k": top_k or self.default_top_k,
            "prediction_top_k": prediction_top_k,
            "warnings": [],
            "timings": {},
        }
        return self._chain.invoke(state)

    def _predict_step(self, state: dict) -> dict:
        started = time.perf_counter()
        prediction = self.inference_engine.predict_bytes(
            state["image_bytes"],
            top_k=state["prediction_top_k"],
        )
        logger.info(
            "Prediction complete label=%s confidence=%.3f",
            prediction.predicted_label,
            prediction.predicted_confidence,
        )
        state["prediction"] = prediction
        state["timings"]["classification_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if prediction.warning:
            state["warnings"].append(prediction.warning)
        return state

    def _build_query_step(self, state: dict) -> dict:
        prediction: PredictionResult = state["prediction"]
        rag_query = build_initial_rag_query(
            predicted_label=prediction.predicted_label,
            top_k=state["top_k"],
        )
        logger.info(
            "Built initial RAG query label=%s disease_tag=%s top_k=%s",
            prediction.predicted_label,
            rag_query.disease_tag,
            rag_query.top_k,
        )
        state["rag_query"] = rag_query
        return state

    def _answer_step(self, state: dict) -> dict:
        started = time.perf_counter()
        rag_query: BuiltRagQuery = state["rag_query"]
        payload = MedicalQueryInput(
            question=rag_query.question,
            disease_tag=rag_query.disease_tag,
            top_k=rag_query.top_k,
        )
        try:
            rag_result = self.answer_fn(payload, config=self.rag_config)
        except Exception as exc:
            logger.warning("Initial literature synthesis failed: %s", exc, exc_info=True)
            state["warnings"].append(f"Literature synthesis unavailable: {exc}")
            rag_result = {
                "answer": "Prediction completed, but literature synthesis is currently unavailable.",
                "citations": [],
            }
        state["timings"]["rag_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "prediction": state["prediction"].to_dict(),
            "initial_summary": {"text": rag_result["answer"]},
            "rag_query": {
                "question": rag_query.question,
                "disease_tag": rag_query.disease_tag,
                "top_k": rag_query.top_k,
            },
            "citations": rag_result.get("citations", []),
            "timings": state["timings"],
            "warnings": state["warnings"],
        }


class FollowupQuestionWorkflow:
    def __init__(
        self,
        *,
        known_labels: list[str],
        rag_config: RAGConfig,
        default_top_k: int,
        answer_fn: Callable[..., dict] = answer_with_citations,
    ) -> None:
        self.known_labels = known_labels
        self.rag_config = rag_config
        self.default_top_k = default_top_k
        self.answer_fn = answer_fn

    def run(
        self,
        *,
        condition: str,
        question: str,
        top_k: int | None = None,
    ) -> dict:
        started = time.perf_counter()
        rag_query = build_followup_rag_query(
            condition=condition,
            question=question,
            top_k=top_k or self.default_top_k,
            known_labels=self.known_labels,
        )
        warnings: list[str] = []
        payload = MedicalQueryInput(
            question=rag_query.question,
            disease_tag=rag_query.disease_tag,
            top_k=rag_query.top_k,
        )
        try:
            rag_result = self.answer_fn(payload, config=self.rag_config)
        except Exception as exc:
            logger.warning("Follow-up literature synthesis failed: %s", exc, exc_info=True)
            warnings.append(f"Literature synthesis unavailable: {exc}")
            rag_result = {
                "answer": "Follow-up answer is currently unavailable.",
                "citations": [],
            }
        return {
            "condition": _normalize_label(condition),
            "question": rag_query.question,
            "answer": {"text": rag_result["answer"]},
            "citations": rag_result.get("citations", []),
            "timings": {"rag_ms": round((time.perf_counter() - started) * 1000, 2)},
            "warnings": warnings,
        }


def _build_chain(*steps):
    try:
        runnable_module = require_module("langchain_core.runnables")
    except RuntimeError:
        return _SequentialChain(steps)

    runnable_cls = getattr(runnable_module, "RunnableLambda")
    chain = runnable_cls(steps[0])
    for step in steps[1:]:
        chain = chain | runnable_cls(step)
    return chain


class _SequentialChain:
    def __init__(self, steps):
        self.steps = steps

    def invoke(self, state: dict) -> dict:
        current = state
        for step in self.steps:
            current = step(current)
        return current
