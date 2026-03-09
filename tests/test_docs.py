from __future__ import annotations

from pathlib import Path


def test_llamaindex_doc_references_expected_sections_and_paths():
    doc_path = Path("rag/README.md")
    text = doc_path.read_text(encoding="utf-8")

    assert "# LlamaIndex + Chroma RAG Guide" in text
    assert "## Source of Truth" in text
    assert "## How to Build and Inspect the Index" in text
    assert "## How to Query the Index" in text
    assert "## Configuration" in text
    assert "## Retrieval Behavior" in text
    assert "## How to Modify This Layer" in text
    assert "make build-index" in text
    assert "make inspect-index" in text
    assert "python scripts/index_articles.py build-index" in text
    assert "python scripts/index_articles.py inspect-index" in text

    for path in [
        Path("rag/config.py"),
        Path("rag/index_builder.py"),
        Path("rag/parsing.py"),
        Path("rag/chunking.py"),
        Path("rag/retrieval.py"),
        Path("rag/agent_tools.py"),
    ]:
        assert path.exists(), f"Referenced path does not exist: {path}"
