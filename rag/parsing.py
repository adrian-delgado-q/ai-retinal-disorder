from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._deps import require_module
from .metadata import extract_year_from_pdf_metadata


REFERENCE_HEADER_PATTERN = re.compile(
    r"(?im)^(references|bibliography|acknowledg(?:e)?ments?|funding|conflicts? of interest|declarations?)\s*$"
)


@dataclass(frozen=True)
class ParsedPdf:
    text: str
    page_count: int
    metadata: dict[str, Any]
    metadata_year: int | None


def clean_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", stripped)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def truncate_reference_tail(text: str) -> str:
    if len(text) < 2000:
        return text
    match = REFERENCE_HEADER_PATTERN.search(text)
    if not match or match.start() < int(len(text) * 0.45):
        return text
    return text[: match.start()].rstrip()


def extract_pdf_text(pdf_path: Path) -> ParsedPdf:
    fitz = require_module("fitz")
    document = fitz.open(pdf_path)
    metadata = dict(document.metadata or {})
    pages = [document.load_page(page_number).get_text("text") for page_number in range(document.page_count)]
    cleaned_text = truncate_reference_tail(clean_extracted_text("\n\n".join(pages)))
    return ParsedPdf(
        text=cleaned_text,
        page_count=document.page_count,
        metadata=metadata,
        metadata_year=extract_year_from_pdf_metadata(metadata),
    )
