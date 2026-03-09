from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app_config import RAGSettings, load_config


DEFAULT_DATASET_PATH = Path("data/articles/eye_conditions_open_access_database.json")
DEFAULT_CHROMA_DIR = Path("data/articles/chroma")
DEFAULT_CHROMA_COLLECTION = "medical_articles"
DEFAULT_EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass(frozen=True)
class RAGConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    chroma_dir: Path = DEFAULT_CHROMA_DIR
    chroma_collection: str = DEFAULT_CHROMA_COLLECTION
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME
    deepseek_api_key: str | None = None
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    deepseek_temperature: float = 0.1
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 150
    embedding_batch_size: int = 16
    default_top_k: int = 5
    retrieval_candidate_multiplier: int = 5
    exclude_low_trust_by_default: bool = True
    excluded_trust_levels: tuple[str, ...] = ("low",)

    @classmethod
    def from_settings(cls, settings: RAGSettings, **overrides) -> "RAGConfig":
        config = cls(**settings.model_dump())
        for key, value in overrides.items():
            if value is not None:
                config = replace(config, **{key: value})
        return config

    @classmethod
    def from_env(cls, **overrides) -> "RAGConfig":
        return cls.from_settings(load_config().rag, **overrides)

    def require_deepseek(self) -> str:
        if not self.deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required for synthesis. "
                "Set it in the environment before using `query-answer` or the synthesis tool."
            )
        return self.deepseek_api_key
