from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ._deps import require_module
from .chunking import chunk_document
from .config import RAGConfig
from .ids import build_doc_id, build_node_id
from .metadata import (
    ArticleRecord,
    build_node_metadata,
    derive_year,
    deserialize_metadata_from_chroma,
    is_downloadable_record,
    load_article_records,
    metadata_missing_keys,
    serialize_metadata_for_chroma,
)
from .parsing import ParsedPdf, extract_pdf_text


EmbedBatchFn = Callable[[list[str]], list[list[float]]]
ParserFn = Callable[[Path], ParsedPdf]


@dataclass(frozen=True)
class PreparedNode:
    node_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class BuildReport:
    dataset_path: str
    processed_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    total_chunks: int = 0
    missing_files: list[str] | None = None
    failures: list[dict[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.missing_files is None:
            self.missing_files = []
        if self.failures is None:
            self.failures = []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_embedder(config: RAGConfig):
    module = require_module("llama_index.embeddings.huggingface")
    embedder_cls = getattr(module, "HuggingFaceEmbedding")
    return embedder_cls(model_name=config.embed_model_name)


def _get_chroma_collection(config: RAGConfig):
    chromadb = require_module("chromadb")
    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    return client.get_or_create_collection(
        name=config.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(embedder, texts: list[str]) -> list[list[float]]:
    return embedder.get_text_embedding_batch(texts)


def prepare_nodes_for_record(
    record: ArticleRecord,
    config: RAGConfig,
    *,
    parser: ParserFn = extract_pdf_text,
) -> list[PreparedNode]:
    parsed = parser(Path(record.path))
    year = derive_year(record, pdf_metadata_year=parsed.metadata_year)
    doc_id = build_doc_id(record.doi, record.article_index, record.path)
    chunks = chunk_document(
        parsed.text,
        chunk_size_chars=config.chunk_size_chars,
        chunk_overlap_chars=config.chunk_overlap_chars,
    )
    prepared_nodes: list[PreparedNode] = []
    for chunk in chunks:
        metadata = build_node_metadata(
            record,
            year=year,
            section=chunk.section,
            extra={"doc_id": doc_id, "chunk_index": chunk.chunk_index},
        )
        prepared_nodes.append(
            PreparedNode(
                node_id=build_node_id(doc_id, chunk.section, chunk.chunk_index, chunk.text),
                doc_id=doc_id,
                text=chunk.text,
                metadata=metadata,
            )
        )
    return prepared_nodes


def upsert_prepared_nodes(
    collection,
    nodes: Iterable[PreparedNode],
    *,
    embed_batch_fn: EmbedBatchFn,
    batch_size: int,
) -> int:
    prepared = list(nodes)
    total = 0
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        texts = [node.text for node in batch]
        embeddings = embed_batch_fn(texts)
        collection.upsert(
            ids=[node.node_id for node in batch],
            documents=texts,
            metadatas=[serialize_metadata_for_chroma(node.metadata) for node in batch],
            embeddings=embeddings,
        )
        total += len(batch)
    return total


def build_index(
    config: RAGConfig | None = None,
    *,
    collection=None,
    embedder=None,
    parser: ParserFn = extract_pdf_text,
) -> BuildReport:
    config = config or RAGConfig.from_env()
    records = load_article_records(config.dataset_path)
    report = BuildReport(dataset_path=str(config.dataset_path))

    if collection is None:
        collection = _get_chroma_collection(config)
    if embedder is None:
        embedder = _get_embedder(config)

    for record in records:
        if not is_downloadable_record(record):
            report.skipped_records += 1
            continue
        if not Path(record.path).exists():
            report.skipped_records += 1
            report.missing_files.append(record.path)
            continue
        try:
            nodes = prepare_nodes_for_record(record, config, parser=parser)
            chunk_count = upsert_prepared_nodes(
                collection,
                nodes,
                embed_batch_fn=lambda texts: _embed_texts(embedder, texts),
                batch_size=config.embedding_batch_size,
            )
            report.processed_records += 1
            report.total_chunks += chunk_count
        except Exception as exc:  # pragma: no cover - integration path
            report.failed_records += 1
            report.failures.append(
                {
                    "article_index": str(record.article_index),
                    "title": record.title,
                    "path": record.path,
                    "error": str(exc),
                }
            )
    return report


def summarize_dataset(records: Iterable[ArticleRecord]) -> dict[str, Any]:
    records = list(records)
    missing_files = [
        record.path
        for record in records
        if record.download_status == "downloaded" and record.path and not Path(record.path).exists()
    ]
    return {
        "record_count": len(records),
        "downloaded_count": sum(1 for record in records if record.download_status == "downloaded"),
        "failed_download_count": sum(1 for record in records if record.download_status != "downloaded"),
        "trust_level_distribution": dict(Counter(record.trust_level for record in records)),
        "source_distribution": dict(Counter(record.source for record in records)),
        "missing_file_count": len(missing_files),
        "missing_files": missing_files,
    }


def inspect_index(config: RAGConfig | None = None) -> dict[str, Any]:
    config = config or RAGConfig.from_env()
    records = load_article_records(config.dataset_path)
    summary = summarize_dataset(records)
    summary["dataset_path"] = str(config.dataset_path)

    try:
        collection = _get_chroma_collection(config)
    except RuntimeError as exc:
        summary.update({"index_ready": False, "dependency_error": str(exc)})
        return summary

    chunk_count = collection.count()
    result = collection.get(include=["metadatas"], limit=chunk_count) if chunk_count else {"metadatas": []}
    metadata_rows = [deserialize_metadata_from_chroma(item) for item in result.get("metadatas", [])]
    missing_key_counter: Counter[str] = Counter()
    for metadata in metadata_rows:
        missing_key_counter.update(metadata_missing_keys(metadata))

    summary.update(
        {
            "index_ready": True,
            "chroma_dir": str(config.chroma_dir),
            "chroma_collection": config.chroma_collection,
            "chunk_count": chunk_count,
            "indexed_document_count": len({item.get("doc_id") for item in metadata_rows if item.get("doc_id")}),
            "metadata_missing_key_counts": dict(missing_key_counter),
            "indexed_trust_level_distribution": dict(
                Counter(item.get("trust_level") for item in metadata_rows if item.get("trust_level"))
            ),
            "indexed_source_distribution": dict(
                Counter(item.get("source") for item in metadata_rows if item.get("source"))
            ),
        }
    )
    return summary
