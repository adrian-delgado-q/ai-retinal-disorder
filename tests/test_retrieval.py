from __future__ import annotations

from rag.config import RAGConfig
from rag.retrieval import RetrievalFilters, query_raw
from tests.test_index_builder import FakeCollection, FakeEmbedder


def _seed_collection():
    collection = FakeCollection()
    embedder = FakeEmbedder()
    texts = [
        "Glaucoma risk factors include age and elevated intraocular pressure.",
        "Myopia prevalence has increased over recent decades.",
        "Older glaucoma cohorts were described in 2018.",
    ]
    collection.upsert(
        ids=["node:1", "node:2", "node:3"],
        documents=texts,
        metadatas=[
            {
                "title": "Glaucoma Paper",
                "source": "PMC",
                "trust_level": "high",
                "path": "paper1.pdf",
                "disease_tags": '["glaucoma"]',
                "primary_disease_tag": "glaucoma",
                "year": 2022,
                "section": "Discussion",
                "article_index": 1,
                "doi": "",
                "url": "",
                "doc_id": "article:1",
                "chunk_index": 0,
            },
            {
                "title": "Myopia Paper",
                "source": "PMC",
                "trust_level": "high",
                "path": "paper2.pdf",
                "disease_tags": '["myopia"]',
                "primary_disease_tag": "myopia",
                "year": 2024,
                "section": "Introduction",
                "article_index": 2,
                "doi": "",
                "url": "",
                "doc_id": "article:2",
                "chunk_index": 0,
            },
            {
                "title": "Low Trust Glaucoma",
                "source": "PMC",
                "trust_level": "low",
                "path": "paper3.pdf",
                "disease_tags": '["glaucoma"]',
                "primary_disease_tag": "glaucoma",
                "year": 2018,
                "section": "Results",
                "article_index": 3,
                "doi": "",
                "url": "",
                "doc_id": "article:3",
                "chunk_index": 0,
            },
        ],
        embeddings=embedder.get_text_embedding_batch(texts),
    )
    return collection, embedder


def test_query_raw_applies_filters_and_default_low_trust_exclusion():
    collection, embedder = _seed_collection()
    config = RAGConfig.from_env()
    filters = RetrievalFilters(disease_tag="glaucoma", source="PMC", year_min=2020, year_max=2023, top_k=5)
    results = query_raw("glaucoma", filters, config, collection=collection, embedder=embedder)
    assert len(results) == 1
    assert results[0]["metadata"]["title"] == "Glaucoma Paper"
    assert collection.last_query["where"] is not None


def test_query_raw_can_include_low_trust_when_requested():
    collection, embedder = _seed_collection()
    config = RAGConfig.from_env()
    filters = RetrievalFilters(disease_tag="glaucoma", top_k=5, allow_low_trust=True)
    results = query_raw("glaucoma", filters, config, collection=collection, embedder=embedder)
    titles = {item["metadata"]["title"] for item in results}
    assert "Low Trust Glaucoma" in titles
