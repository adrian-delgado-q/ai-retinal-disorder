from __future__ import annotations

from rag.chunking import chunk_document, split_into_sections


def test_split_into_sections_detects_common_headings():
    text = "Abstract\n\nShort abstract.\n\nIntroduction\n\nIntro body.\n\nResults\n\nResult body."
    sections = split_into_sections(text)
    assert sections[0][0] == "Abstract"
    assert sections[1][0] == "Introduction"
    assert sections[2][0] == "Results"


def test_chunk_document_emits_non_empty_chunks_with_indices():
    text = (
        "Introduction\n\nSentence one. Sentence two. Sentence three. Sentence four.\n\n"
        "Discussion\n\nSentence five. Sentence six."
    )
    chunks = chunk_document(text, chunk_size_chars=40, chunk_overlap_chars=10)
    assert chunks
    assert all(chunk.text for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
