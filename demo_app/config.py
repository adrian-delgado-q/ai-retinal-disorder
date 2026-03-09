from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app_config import DemoSettings, load_config
from rag.config import RAGConfig


@dataclass(frozen=True)
class DemoAppConfig:
    model_artifact_source_url: str | None = None
    model_artifact_local_dir: Path = Path("data/model_artifacts")
    model_artifact_force_refresh: bool = False
    model_artifact_run_subdir: str | None = None
    classifier_device: str = "cpu"
    classifier_low_conf_threshold: float = 0.55
    classifier_top_k: int = 3
    rag_top_k: int = 3
    app_title: str = "Retinal Image + RAG Demo"
    api_host: str = "127.0.0.1"
    api_port: int = 8001

    @classmethod
    def from_settings(cls, settings: DemoSettings | None = None) -> "DemoAppConfig":
        app_config = load_config()
        settings = settings or app_config.demo
        return cls(
            model_artifact_source_url=settings.model_artifact_source_url,
            model_artifact_local_dir=settings.model_artifact_local_dir,
            model_artifact_force_refresh=settings.model_artifact_force_refresh,
            model_artifact_run_subdir=settings.model_artifact_run_subdir,
            classifier_device=settings.classifier_device,
            classifier_low_conf_threshold=settings.classifier_low_conf_threshold,
            classifier_top_k=settings.classifier_top_k,
            rag_top_k=app_config.effective_demo_rag_top_k,
            app_title=settings.app_title,
            api_host=settings.api_host,
            api_port=settings.api_port,
        )

    @classmethod
    def from_env(cls) -> "DemoAppConfig":
        return cls.from_settings()

    def build_rag_config(self) -> RAGConfig:
        return RAGConfig.from_env(default_top_k=self.rag_top_k)
