from __future__ import annotations

from rag.ids import build_doc_id, build_node_id


def test_build_doc_id_prefers_doi():
    assert build_doc_id("10.1000/XYZ-12", 3, "paper.pdf") == "doi:10-1000-xyz-12"


def test_build_doc_id_falls_back_to_article_index():
    assert build_doc_id(None, 7, "paper.pdf") == "article:7"


def test_build_node_id_is_deterministic():
    node_id_1 = build_node_id("article:1", "Introduction", 0, "Same chunk text")
    node_id_2 = build_node_id("article:1", "Introduction", 0, "Same chunk text")
    assert node_id_1 == node_id_2
