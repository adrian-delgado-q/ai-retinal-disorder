from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Mapping

from ._deps import require_module
from .config import RAGConfig
from .index_builder import _get_chroma_collection, _get_embedder
from .metadata import deserialize_metadata_from_chroma
from runtime_logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalFilters:
    disease_tag: str | None = None
    trust_levels: list[str] | None = None
    source: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    top_k: int = 5
    allow_low_trust: bool = False


def _build_chroma_where(filters: RetrievalFilters, config: RAGConfig) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if filters.source:
        clauses.append({"source": {"$eq": filters.source}})
    if filters.disease_tag:
        clauses.append({"primary_disease_tag": {"$eq": filters.disease_tag}})
    if filters.trust_levels:
        clauses.append({"trust_level": {"$in": filters.trust_levels}})
    elif config.exclude_low_trust_by_default and not filters.allow_low_trust and len(config.excluded_trust_levels) == 1:
        clauses.append({"trust_level": {"$ne": config.excluded_trust_levels[0]}})
    if filters.year_min is not None:
        clauses.append({"year": {"$gte": filters.year_min}})
    if filters.year_max is not None:
        clauses.append({"year": {"$lte": filters.year_max}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _apply_post_filters(
    chunks: Iterable[dict[str, Any]],
    filters: RetrievalFilters,
    config: RAGConfig,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in chunks:
        metadata = item["metadata"]
        if filters.disease_tag and filters.disease_tag not in metadata.get("disease_tags", []):
            continue
        if filters.trust_levels and metadata.get("trust_level") not in filters.trust_levels:
            continue
        if (
            config.exclude_low_trust_by_default
            and not filters.allow_low_trust
            and not filters.trust_levels
            and metadata.get("trust_level") in config.excluded_trust_levels
        ):
            continue
        if filters.year_min is not None and metadata.get("year") is not None and metadata["year"] < filters.year_min:
            continue
        if filters.year_max is not None and metadata.get("year") is not None and metadata["year"] > filters.year_max:
            continue
        filtered.append(item)
    return filtered


def query_raw(
    question: str,
    filters: RetrievalFilters,
    config: RAGConfig | None = None,
    *,
    collection=None,
    embedder=None,
) -> list[dict[str, Any]]:
    config = config or RAGConfig.from_env()
    logger.info(
        "RAG query_raw question=%r disease_tag=%s source=%s top_k=%s allow_low_trust=%s",
        question,
        filters.disease_tag,
        filters.source,
        filters.top_k,
        filters.allow_low_trust,
    )
    if collection is None:
        collection = _get_chroma_collection(config)
    if embedder is None:
        embedder = _get_embedder(config)

    where = _build_chroma_where(filters, config)
    n_results = max(filters.top_k, filters.top_k * config.retrieval_candidate_multiplier)
    query_embedding = embedder.get_query_embedding(question)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items: list[dict[str, Any]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        decoded = deserialize_metadata_from_chroma(metadata)
        items.append(
            {
                "text": text,
                "score": None if distance is None else round(1.0 / (1.0 + distance), 6),
                "metadata": decoded,
            }
        )
    filtered = _apply_post_filters(items, filters, config)
    logger.info("RAG query_raw retrieved=%s filtered=%s", len(items), len(filtered))
    return filtered[: filters.top_k]


def _get_deepseek_llm(config: RAGConfig):
    config.require_deepseek()
    module = require_module("llama_index.llms.openai_like")
    llm_cls = getattr(module, "OpenAILike")
    return llm_cls(
        model=config.deepseek_model,
        api_base=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
        temperature=config.deepseek_temperature,
        is_chat_model=True,
    )


def synthesize_answer(
    question: str,
    filters: RetrievalFilters,
    config: RAGConfig | None = None,
    *,
    collection=None,
    embedder=None,
    llm=None,
) -> dict[str, Any]:
    config = config or RAGConfig.from_env()
    chunks = query_raw(
        question,
        filters,
        config,
        collection=collection,
        embedder=embedder,
    )
    if not chunks:
        logger.info("RAG synthesize_answer found no matching chunks")
        return {"answer": "No indexed context matched the query.", "citations": []}

    if llm is None:
        llm = _get_deepseek_llm(config)
    logger.info("RAG synthesize_answer using %s chunks", len(chunks))

    context_lines = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        context_lines.append(
            (
                f"[{index}] title={metadata.get('title')} | source={metadata.get('source')} "
                f"| year={metadata.get('year')} | section={metadata.get('section')} "
                f"| disease_tags={metadata.get('disease_tags')} | path={metadata.get('path')}\n"
                f"{chunk['text']}"
            )
        )
    prompt = (
        "You are answering questions about ophthalmology literature. "
        "Use only the provided context. If the context is insufficient, say so. "
        "Cite supporting chunks inline using bracketed numbers like [1], [2].\n\n"
        f"Question: {question}\n\nContext:\n" + "\n\n".join(context_lines)
    )
    completion = llm.complete(prompt)
    answer_text = getattr(completion, "text", str(completion))
    logger.info("RAG synthesize_answer completed answer_length=%s", len(answer_text.strip()))
    return {"answer": answer_text.strip(), "citations": chunks}
