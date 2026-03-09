from __future__ import annotations

import json
from pathlib import Path

from rag.config import RAGConfig
from rag.index_builder import build_index
from rag.parsing import ParsedPdf


class FakeEmbedder:
    def get_text_embedding_batch(self, texts):
        return [[float(len(text))] for text in texts]

    def get_query_embedding(self, text):
        return [float(len(text))]


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.last_query = None

    def upsert(self, ids, documents, metadatas, embeddings):
        for row_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.rows[row_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def count(self):
        return len(self.rows)

    def get(self, include=None, limit=None):
        items = list(self.rows.values())
        if limit is not None:
            items = items[:limit]
        return {"metadatas": [item["metadata"] for item in items]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        self.last_query = {"query_embeddings": query_embeddings, "n_results": n_results, "where": where}
        query_vector = query_embeddings[0][0]
        results = []
        for row in self.rows.values():
            metadata = row["metadata"]
            if not _matches_where(metadata, where):
                continue
            distance = abs(query_vector - row["embedding"][0])
            results.append((distance, row))
        results.sort(key=lambda item: item[0])
        trimmed = results[:n_results]
        return {
            "documents": [[item[1]["document"] for item in trimmed]],
            "metadatas": [[item[1]["metadata"] for item in trimmed]],
            "distances": [[item[0] for item in trimmed]],
        }


def _matches_where(metadata, where):
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])
    field, expression = next(iter(where.items()))
    if "$eq" in expression:
        return metadata.get(field) == expression["$eq"]
    if "$ne" in expression:
        return metadata.get(field) != expression["$ne"]
    if "$in" in expression:
        return metadata.get(field) in expression["$in"]
    if "$gte" in expression:
        return metadata.get(field) is not None and metadata.get(field) >= expression["$gte"]
    if "$lte" in expression:
        return metadata.get(field) is not None and metadata.get(field) <= expression["$lte"]
    raise AssertionError(f"Unsupported expression: {expression}")


def fake_parser(_path: Path) -> ParsedPdf:
    return ParsedPdf(
        text="Introduction\n\nSentence one. Sentence two. Sentence three.",
        page_count=1,
        metadata={"creationDate": "D:20230101000000"},
        metadata_year=2023,
    )


def test_build_index_is_idempotent_with_deterministic_ids(tmp_path: Path):
    pdf_path = tmp_path / "article.pdf"
    pdf_path.write_text("placeholder", encoding="utf-8")
    dataset_path = tmp_path / "articles.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "title": "Example Article",
                    "source": "PMC",
                    "disease_tags": ["glaucoma"],
                    "trust_level": "high",
                    "index": 1,
                    "download_status": "downloaded",
                    "path": str(pdf_path),
                }
            ]
        ),
        encoding="utf-8",
    )
    config = RAGConfig.from_env(dataset_path=dataset_path, chroma_dir=tmp_path / "chroma")
    collection = FakeCollection()
    embedder = FakeEmbedder()

    first = build_index(config, collection=collection, embedder=embedder, parser=fake_parser)
    first_count = collection.count()
    second = build_index(config, collection=collection, embedder=embedder, parser=fake_parser)

    assert first.processed_records == 1
    assert second.processed_records == 1
    assert collection.count() == first_count
    assert first_count > 0
