from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ._deps import require_module
from .config import RAGConfig
from .retrieval import RetrievalFilters, query_raw, synthesize_answer


class MedicalQueryInput(BaseModel):
    question: str = Field(..., description="Medical question to answer against the indexed article corpus.")
    disease_tag: str | None = Field(default=None, description="Optional disease tag filter.")
    trust_levels: list[str] | None = Field(default=None, description="Optional explicit trust-level allowlist.")
    source: str | None = Field(default=None, description="Optional source filter.")
    year_min: int | None = Field(default=None, description="Minimum publication year filter.")
    year_max: int | None = Field(default=None, description="Maximum publication year filter.")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of chunks to return.")
    allow_low_trust: bool = Field(
        default=False,
        description="Whether low-trust content can be retrieved even when the baseline filter would exclude it.",
    )


def _filters_from_input(payload: MedicalQueryInput) -> RetrievalFilters:
    return RetrievalFilters(
        disease_tag=payload.disease_tag,
        trust_levels=payload.trust_levels,
        source=payload.source,
        year_min=payload.year_min,
        year_max=payload.year_max,
        top_k=payload.top_k,
        allow_low_trust=payload.allow_low_trust,
    )


def retrieve_medical_chunks(
    payload: MedicalQueryInput | dict[str, Any],
    *,
    config: RAGConfig | None = None,
    collection=None,
    embedder=None,
) -> list[dict[str, Any]]:
    parsed = payload if isinstance(payload, MedicalQueryInput) else MedicalQueryInput(**payload)
    return query_raw(
        parsed.question,
        _filters_from_input(parsed),
        config or RAGConfig.from_env(),
        collection=collection,
        embedder=embedder,
    )


def answer_with_citations(
    payload: MedicalQueryInput | dict[str, Any],
    *,
    config: RAGConfig | None = None,
    collection=None,
    embedder=None,
    llm=None,
) -> dict[str, Any]:
    parsed = payload if isinstance(payload, MedicalQueryInput) else MedicalQueryInput(**payload)
    return synthesize_answer(
        parsed.question,
        _filters_from_input(parsed),
        config or RAGConfig.from_env(),
        collection=collection,
        embedder=embedder,
        llm=llm,
    )


def create_langchain_tools(config: RAGConfig | None = None):
    config = config or RAGConfig.from_env()
    structured_tool_module = require_module("langchain_core.tools")
    StructuredTool = getattr(structured_tool_module, "StructuredTool")

    def _retrieve(**kwargs):
        return retrieve_medical_chunks(kwargs, config=config)

    def _answer(**kwargs):
        return answer_with_citations(kwargs, config=config)

    retrieve_tool = StructuredTool.from_function(
        func=_retrieve,
        name="retrieve_medical_chunks",
        description=(
            "Retrieve raw ophthalmology literature chunks with scores and metadata. "
            "Use this when the agent needs direct evidence before synthesizing."
        ),
        args_schema=MedicalQueryInput,
    )
    answer_tool = StructuredTool.from_function(
        func=_answer,
        name="answer_with_citations",
        description=(
            "Answer a medical literature question using the indexed corpus and return the answer with citations."
        ),
        args_schema=MedicalQueryInput,
    )
    return [retrieve_tool, answer_tool]
