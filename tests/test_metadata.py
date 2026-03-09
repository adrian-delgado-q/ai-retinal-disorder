from __future__ import annotations

from rag.metadata import (
    build_node_metadata,
    derive_year,
    extract_year_from_pdf_metadata,
    normalize_article_record,
)


def test_normalize_article_record_maps_index_to_article_index():
    record = normalize_article_record(
        {
            "title": "Example Article",
            "source": "PMC",
            "disease_tags": ["glaucoma"],
            "trust_level": "high",
            "index": 12,
            "path": "data/articles/downloads/example.pdf",
        }
    )
    assert record.article_index == 12
    assert record.path == "data/articles/downloads/example.pdf"


def test_extract_year_prefers_pdf_metadata():
    assert extract_year_from_pdf_metadata({"creationDate": "D:20240304000000"}) == 2024


def test_derive_year_falls_back_to_title_or_path():
    record = normalize_article_record(
        {
            "title": "Glaucoma Review 2021",
            "source": "PMC",
            "disease_tags": ["glaucoma"],
            "trust_level": "high",
            "index": 2,
            "path": "downloads/paper_2019.pdf",
        }
    )
    assert derive_year(record, pdf_metadata_year=2024) == 2024
    assert derive_year(record, pdf_metadata_year=None) == 2021


def test_build_node_metadata_contains_required_contract():
    record = normalize_article_record(
        {
            "title": "Test",
            "source": "PMC",
            "disease_tags": ["myopia"],
            "trust_level": "high",
            "index": 8,
            "path": "downloads/test.pdf",
        }
    )
    metadata = build_node_metadata(record, year=None, section="Methods")
    assert metadata["title"] == "Test"
    assert metadata["source"] == "PMC"
    assert metadata["disease_tags"] == ["myopia"]
    assert metadata["year"] is None
    assert metadata["section"] == "Methods"
