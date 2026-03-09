from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_NODE_KEYS = (
    "title",
    "source",
    "disease_tags",
    "trust_level",
    "year",
    "section",
    "article_index",
    "doi",
    "url",
    "path",
)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
PDF_DATE_PATTERN = re.compile(r"D:(19\d{2}|20\d{2})")


@dataclass(frozen=True)
class ArticleRecord:
    title: str
    source: str
    disease_tags: list[str]
    trust_level: str
    url: str | None
    article_index: int | None
    doi: str | None
    path: str
    download_status: str | None = None
    pdf_url: str | None = None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def normalize_article_record(payload: Mapping[str, Any]) -> ArticleRecord:
    title = _coerce_optional_str(payload.get("title"))
    source = _coerce_optional_str(payload.get("source"))
    trust_level = _coerce_optional_str(payload.get("trust_level"))
    disease_tags = payload.get("disease_tags")
    if not title or not source or not trust_level:
        raise ValueError(f"Invalid article record: missing title/source/trust_level in {payload}")
    if not isinstance(disease_tags, list) or not all(isinstance(tag, str) for tag in disease_tags):
        raise ValueError(f"Invalid article record: disease_tags must be a list[str] in {payload}")
    return ArticleRecord(
        title=title,
        source=source,
        disease_tags=[tag.strip() for tag in disease_tags if tag.strip()],
        trust_level=trust_level,
        url=_coerce_optional_str(payload.get("url")),
        article_index=_coerce_optional_int(payload.get("article_index", payload.get("index"))),
        doi=_coerce_optional_str(payload.get("doi")),
        path=_coerce_optional_str(payload.get("path")) or "",
        download_status=_coerce_optional_str(payload.get("download_status")),
        pdf_url=_coerce_optional_str(payload.get("pdf_url")),
    )


def load_article_records(dataset_path: Path) -> list[ArticleRecord]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {dataset_path}")
    return [normalize_article_record(item) for item in payload if isinstance(item, Mapping)]


def is_downloadable_record(record: ArticleRecord) -> bool:
    return record.download_status == "downloaded" and bool(record.path)


def extract_year_from_pdf_metadata(metadata: Mapping[str, Any] | None) -> int | None:
    if not metadata:
        return None
    for key in ("creationDate", "modDate", "subject", "title"):
        value = str(metadata.get(key, ""))
        match = PDF_DATE_PATTERN.search(value) or YEAR_PATTERN.search(value)
        if match:
            return int(match.group(1))
    return None


def extract_year_from_text_candidates(candidates: Iterable[str | None]) -> int | None:
    for candidate in candidates:
        if not candidate:
            continue
        match = YEAR_PATTERN.search(candidate)
        if match:
            return int(match.group(1))
    return None


def derive_year(
    record: ArticleRecord,
    *,
    pdf_metadata_year: int | None = None,
) -> int | None:
    if pdf_metadata_year is not None:
        return pdf_metadata_year
    return extract_year_from_text_candidates((record.title, record.path, record.url, record.pdf_url))


def build_node_metadata(
    record: ArticleRecord,
    *,
    year: int | None,
    section: str | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "title": record.title,
        "source": record.source,
        "disease_tags": list(record.disease_tags),
        "trust_level": record.trust_level,
        "year": year,
        "section": section,
        "article_index": record.article_index,
        "doi": record.doi,
        "url": record.url,
        "path": record.path,
    }
    if extra:
        metadata.update(extra)
    return metadata


def serialize_metadata_for_chroma(metadata: Mapping[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {
        "title": metadata["title"],
        "source": metadata["source"],
        "trust_level": metadata["trust_level"],
        "path": metadata["path"],
        "disease_tags": json.dumps(metadata["disease_tags"]),
        "primary_disease_tag": metadata["disease_tags"][0] if metadata["disease_tags"] else "",
        "section": metadata.get("section") or "",
        "doi": metadata.get("doi") or "",
        "url": metadata.get("url") or "",
    }
    if metadata.get("year") is not None:
        serialized["year"] = int(metadata["year"])
    if metadata.get("article_index") is not None:
        serialized["article_index"] = int(metadata["article_index"])
    for key in ("doc_id", "chunk_index"):
        if key in metadata and metadata[key] is not None:
            serialized[key] = metadata[key]
    return serialized


def deserialize_metadata_from_chroma(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(metadata or {})
    disease_tags_raw = metadata.get("disease_tags", "[]")
    try:
        disease_tags = json.loads(disease_tags_raw)
    except json.JSONDecodeError:
        disease_tags = []
    decoded = {
        "title": metadata.get("title", ""),
        "source": metadata.get("source", ""),
        "disease_tags": disease_tags if isinstance(disease_tags, list) else [],
        "trust_level": metadata.get("trust_level", ""),
        "year": metadata.get("year"),
        "section": metadata.get("section") or None,
        "article_index": metadata.get("article_index"),
        "doi": metadata.get("doi") or None,
        "url": metadata.get("url") or None,
        "path": metadata.get("path", ""),
    }
    for key in ("doc_id", "chunk_index"):
        if key in metadata:
            decoded[key] = metadata[key]
    return decoded


def metadata_missing_keys(metadata: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in REQUIRED_NODE_KEYS:
        if key not in metadata:
            missing.append(key)
    return missing
