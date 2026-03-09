from __future__ import annotations

from dataclasses import replace

from rag.agent_tools import MedicalQueryInput, answer_with_citations, retrieve_medical_chunks
from rag.config import RAGConfig
from rag.retrieval import query_raw
from tests.test_index_builder import FakeCollection, FakeEmbedder


def _seed_collection():
    collection = FakeCollection()
    embedder = FakeEmbedder()
    collection.upsert(
        ids=["node:1", "node:2"],
        documents=[
            "Glaucoma risk factors include age and intraocular pressure.",
            "Myopia is associated with axial elongation.",
        ],
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
        ],
        embeddings=embedder.get_text_embedding_batch(
            [
                "Glaucoma risk factors include age and intraocular pressure.",
                "Myopia is associated with axial elongation.",
            ]
        ),
    )
    return collection, embedder


def test_medical_query_input_validates_defaults():
    payload = MedicalQueryInput(question="What is glaucoma?")
    assert payload.top_k == 5
    assert payload.allow_low_trust is False


def test_retrieve_medical_chunks_returns_filtered_results():
    collection, embedder = _seed_collection()
    config = RAGConfig.from_env()
    payload = MedicalQueryInput(question="glaucoma", disease_tag="glaucoma", top_k=2)
    results = retrieve_medical_chunks(payload, config=config, collection=collection, embedder=embedder)
    assert len(results) == 1
    assert results[0]["metadata"]["disease_tags"] == ["glaucoma"]


def test_answer_with_citations_raises_without_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    collection, embedder = _seed_collection()
    config = replace(RAGConfig.from_env(), deepseek_api_key=None)
    payload = MedicalQueryInput(question="Summarize glaucoma findings", top_k=1)
    try:
        answer_with_citations(payload, config=config, collection=collection, embedder=embedder)
    except RuntimeError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected RuntimeError when DEEPSEEK_API_KEY is missing")
