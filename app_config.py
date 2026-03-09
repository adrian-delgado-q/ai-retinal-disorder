from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _settings_config() -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SettingsBase(BaseSettings):
    model_config = _settings_config()


class TrainingSettings(SettingsBase):
    model_name: str = Field("efficientnet_b0", alias="TRAIN_MODEL")
    image_size: int = Field(224, alias="TRAIN_IMAGE_SIZE")
    batch_size: int = Field(32, alias="TRAIN_BATCH_SIZE")
    epochs: int = Field(15, alias="TRAIN_EPOCHS")
    learning_rate: float = Field(3e-4, alias="TRAIN_LR")
    weight_decay: float = Field(1e-4, alias="TRAIN_WEIGHT_DECAY")
    num_workers: int = Field(2, alias="TRAIN_NUM_WORKERS")
    patience: int = Field(4, alias="TRAIN_PATIENCE")
    seed: int = Field(42, alias="TRAIN_SEED")
    output_dir: Path = Field(Path("runs/default"), alias="TRAIN_OUTPUT_DIR")
    data_root: Path | None = Field(None, alias="TRAIN_DATA_ROOT")
    train_csv: Path = Field(Path("data/splits/train_clean.csv"), alias="TRAIN_TRAIN_CSV")
    val_csv: Path = Field(Path("data/splits/val_clean.csv"), alias="TRAIN_VAL_CSV")
    test_csv: Path = Field(Path("data/splits/test_clean.csv"), alias="TRAIN_TEST_CSV")
    provisional_train_csv: Path = Field(Path("data/splits/train.csv"), alias="TRAIN_PROVISIONAL_TRAIN_CSV")
    provisional_val_csv: Path = Field(Path("data/splits/val.csv"), alias="TRAIN_PROVISIONAL_VAL_CSV")
    provisional_test_csv: Path = Field(Path("data/splits/test.csv"), alias="TRAIN_PROVISIONAL_TEST_CSV")
    allow_provisional_data: bool = Field(False, alias="TRAIN_ALLOW_PROVISIONAL_DATA")
    run_test_after_training: bool = Field(False, alias="TRAIN_RUN_TEST_AFTER_TRAINING")


class RAGSettings(SettingsBase):
    dataset_path: Path = Field(Path("data/articles/eye_conditions_open_access_database.json"), alias="ARTICLE_DATASET_PATH")
    chroma_dir: Path = Field(Path("data/articles/chroma"), alias="CHROMA_DIR")
    chroma_collection: str = Field("medical_articles", alias="CHROMA_COLLECTION")
    embed_model_name: str = Field("pritamdeka/S-PubMedBert-MS-MARCO", alias="EMBED_MODEL_NAME")
    deepseek_api_key: str | None = Field(
        None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "DEEPSEEK_TOKEN"),
    )
    deepseek_model: str = Field("deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field("https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_temperature: float = Field(0.1, alias="DEEPSEEK_TEMPERATURE")
    chunk_size_chars: int = Field(1200, alias="RAG_CHUNK_SIZE")
    chunk_overlap_chars: int = Field(150, alias="RAG_CHUNK_OVERLAP")
    embedding_batch_size: int = Field(16, alias="RAG_EMBED_BATCH_SIZE")
    default_top_k: int = Field(5, alias="RAG_TOP_K")
    retrieval_candidate_multiplier: int = Field(5, alias="RAG_CANDIDATE_MULTIPLIER")
    exclude_low_trust_by_default: bool = Field(True, alias="RAG_EXCLUDE_LOW_TRUST")
    excluded_trust_levels: tuple[str, ...] = Field(("low",), alias="RAG_EXCLUDED_TRUST_LEVELS")

    @field_validator("excluded_trust_levels", mode="before")
    @classmethod
    def _normalize_trust_levels(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ("low",)
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(part).strip() for part in value if str(part).strip())
        raise TypeError("excluded_trust_levels must be a comma-delimited string or sequence")


class DemoSettings(SettingsBase):
    model_artifact_source_url: str | None = Field(None, alias="MODEL_ARTIFACT_SOURCE_URL")
    model_artifact_local_dir: Path = Field(Path("data/model_artifacts"), alias="MODEL_ARTIFACT_LOCAL_DIR")
    model_artifact_force_refresh: bool = Field(False, alias="MODEL_ARTIFACT_FORCE_REFRESH")
    model_artifact_run_subdir: str | None = Field(None, alias="MODEL_ARTIFACT_RUN_SUBDIR")
    classifier_device: str = Field("cpu", alias="CLASSIFIER_DEVICE")
    classifier_low_conf_threshold: float = Field(0.55, alias="CLASSIFIER_LOW_CONF_THRESHOLD")
    classifier_top_k: int = Field(4, alias="CLASSIFIER_TOP_K")
    rag_top_k: int | None = Field(None, alias="APP_RAG_TOP_K")
    app_title: str = Field("Retinal Image + RAG Demo", alias="APP_TITLE")
    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8001, alias="API_PORT")


class FrontendSettings(SettingsBase):
    demo_api_url: str | None = Field(None, alias="DEMO_API_URL")


@dataclass(frozen=True)
class AppConfig:
    training: TrainingSettings
    rag: RAGSettings
    demo: DemoSettings
    frontend: FrontendSettings

    @property
    def effective_demo_rag_top_k(self) -> int:
        return self.demo.rag_top_k or self.rag.default_top_k

    @property
    def effective_frontend_api_url(self) -> str:
        return self.frontend.demo_api_url or f"http://{self.demo.api_host}:{self.demo.api_port}"

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        payload = {
            "training": self.training.model_dump(mode="json"),
            "rag": self.rag.model_dump(mode="json"),
            "demo": self.demo.model_dump(mode="json"),
            "frontend": self.frontend.model_dump(mode="json"),
            "derived": {
                "effective_demo_rag_top_k": self.effective_demo_rag_top_k,
                "effective_frontend_api_url": self.effective_frontend_api_url,
            },
        }
        if mask_secrets and payload["rag"].get("deepseek_api_key"):
            payload["rag"]["deepseek_api_key"] = "***"
        return payload

    def to_json(self, *, mask_secrets: bool = True) -> str:
        return json.dumps(self.to_dict(mask_secrets=mask_secrets), indent=2, sort_keys=True, default=str)


@lru_cache(maxsize=None)
def load_config(env_file: str | Path | None = ".env") -> AppConfig:
    env_file_arg = str(env_file) if env_file is not None else None
    return AppConfig(
        training=TrainingSettings(_env_file=env_file_arg),
        rag=RAGSettings(_env_file=env_file_arg),
        demo=DemoSettings(_env_file=env_file_arg),
        frontend=FrontendSettings(_env_file=env_file_arg),
    )


def clear_config_cache() -> None:
    load_config.cache_clear()
