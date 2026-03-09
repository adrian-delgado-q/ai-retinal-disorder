from __future__ import annotations

from rag.parsing import clean_extracted_text, truncate_reference_tail


def test_clean_extracted_text_repairs_hyphenation_and_paragraphs():
    text = "This is hy-\nphenated text.\nStill same paragraph.\n\nSecond paragraph."
    cleaned = clean_extracted_text(text)
    assert "hyphenated text. Still same paragraph." in cleaned
    assert "\n\nSecond paragraph." in cleaned


def test_truncate_reference_tail_when_reference_heading_is_late():
    body = "Intro\n\n" + ("Content. " * 300)
    text = body + "\n\nReferences\n\n[1] Citation"
    truncated = truncate_reference_tail(text)
    assert "References" not in truncated
