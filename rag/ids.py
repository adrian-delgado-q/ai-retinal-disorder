from __future__ import annotations

import hashlib
import re


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def build_doc_id(doi: str | None, article_index: int | None, path: str) -> str:
    if doi:
        normalized = re.sub(r"[^a-z0-9]+", "-", doi.lower()).strip("-")
        return f"doi:{normalized}"
    if article_index is not None:
        return f"article:{article_index}"
    return f"path:{stable_hash(path)}"


def normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_node_id(doc_id: str, section: str | None, chunk_index: int, chunk_text: str) -> str:
    section_key = (section or "body").strip().lower()
    fingerprint = f"{doc_id}|{section_key}|{chunk_index}|{normalize_chunk_text(chunk_text)}"
    return f"node:{stable_hash(fingerprint)}"
