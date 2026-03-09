from __future__ import annotations

import re
from dataclasses import dataclass


SECTION_NAMES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "methods": "Methods",
    "materials and methods": "Methods",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "case report": "Case Report",
}


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str | None
    chunk_index: int


def _normalize_heading(paragraph: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", paragraph.lower()).strip()


def detect_section_heading(paragraph: str) -> str | None:
    normalized = _normalize_heading(paragraph)
    if normalized in SECTION_NAMES:
        return SECTION_NAMES[normalized]
    if (
        paragraph == paragraph.upper()
        and 2 <= len(paragraph.split()) <= 6
        and len(paragraph) <= 80
    ):
        return paragraph.title()
    return None


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        heading = detect_section_heading(paragraph)
        if heading:
            if current_paragraphs:
                sections.append((current_heading, "\n\n".join(current_paragraphs)))
                current_paragraphs = []
            current_heading = heading
            continue
        current_paragraphs.append(paragraph)

    if current_paragraphs:
        sections.append((current_heading, "\n\n".join(current_paragraphs)))
    if not sections and text.strip():
        sections.append((None, text.strip()))
    return sections


def split_into_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if sentence.strip()
    ]
    return sentences or [text.strip()]


def _chunk_long_sentence(sentence: str, chunk_size_chars: int, chunk_overlap_chars: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    step = max(1, chunk_size_chars - chunk_overlap_chars)
    while start < len(sentence):
        pieces.append(sentence[start : start + chunk_size_chars].strip())
        start += step
    return [piece for piece in pieces if piece]


def _chunk_section_text(section_text: str, chunk_size_chars: int, chunk_overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in split_into_sentences(section_text):
        if len(sentence) > chunk_size_chars:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_length = 0
            chunks.extend(_chunk_long_sentence(sentence, chunk_size_chars, chunk_overlap_chars))
            continue

        projected = current_length + len(sentence) + (1 if current_sentences else 0)
        if current_sentences and projected > chunk_size_chars:
            chunks.append(" ".join(current_sentences).strip())
            overlap_sentences: list[str] = []
            overlap_length = 0
            for existing in reversed(current_sentences):
                overlap_sentences.insert(0, existing)
                overlap_length += len(existing) + 1
                if overlap_length >= chunk_overlap_chars:
                    break
            current_sentences = overlap_sentences
            current_length = len(" ".join(current_sentences))

        current_sentences.append(sentence)
        current_length = len(" ".join(current_sentences))

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_document(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for section, section_text in split_into_sections(text):
        for chunk_text in _chunk_section_text(section_text, chunk_size_chars, chunk_overlap_chars):
            chunks.append(Chunk(text=chunk_text, section=section, chunk_index=chunk_index))
            chunk_index += 1
    return chunks
