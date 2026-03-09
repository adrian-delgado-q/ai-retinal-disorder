from __future__ import annotations

from pathlib import Path

from app_config import clear_config_cache, load_config
from demo_app.config import DemoAppConfig
from demo_app.main import DemoRuntime, create_app
from rag.config import RAGConfig
from scripts.index_articles import build_config, parse_args as parse_index_args
from training.train_colab import parse_args as parse_train_args


def _clear(monkeypatch) -> None:
    for name in (
        "TRAIN_BATCH_SIZE",
        "TRAIN_OUTPUT_DIR",
        "CHROMA_DIR",
        "ARTICLE_DATASET_PATH",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_TOKEN",
        "DEEPSEEK_TEMPERATURE",
        "TRAIN_ALLOW_PROVISIONAL_DATA",
        "RAG_TOP_K",
        "APP_RAG_TOP_K",
        "API_PORT",
        "APP_TITLE",
        "DEMO_API_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_config_cache()


def test_load_config_defaults_without_env_file(monkeypatch):
    _clear(monkeypatch)

    config = load_config(env_file=None)

    assert config.training.batch_size == 32
    assert config.rag.default_top_k == 5
    assert config.demo.api_port == 8001
    assert config.effective_frontend_api_url == "http://127.0.0.1:8001"


def test_load_config_reads_env_file_and_aliases(tmp_path: Path, monkeypatch):
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TRAIN_BATCH_SIZE=48",
                "CHROMA_DIR=data/custom-chroma",
                "DEEPSEEK_TOKEN=legacy-secret",
                "DEEPSEEK_TEMPERATURE=0.3",
                "TRAIN_ALLOW_PROVISIONAL_DATA=true",
                "API_PORT=8100",
                "DEMO_API_URL=http://127.0.0.1:8100",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(env_file=env_file)

    assert config.training.batch_size == 48
    assert config.training.allow_provisional_data is True
    assert config.rag.chroma_dir == Path("data/custom-chroma")
    assert config.rag.deepseek_api_key == "legacy-secret"
    assert config.rag.deepseek_temperature == 0.3
    assert config.demo.api_port == 8100
    assert config.frontend.demo_api_url == "http://127.0.0.1:8100"


def test_env_overrides_env_file(monkeypatch, tmp_path: Path):
    _clear(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("TRAIN_BATCH_SIZE=16\nAPI_PORT=8100\n", encoding="utf-8")
    monkeypatch.setenv("TRAIN_BATCH_SIZE", "96")
    monkeypatch.setenv("API_PORT", "8200")

    config = load_config(env_file=env_file)

    assert config.training.batch_size == 96
    assert config.demo.api_port == 8200


def test_rag_config_from_env_keeps_shared_defaults_and_overrides(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("CHROMA_DIR", "data/shared-chroma")
    clear_config_cache()

    args = parse_index_args(["inspect-index", "--dataset-path", "data/override.json"])
    config = build_config(args)

    assert isinstance(config, RAGConfig)
    assert config.dataset_path == Path("data/override.json")
    assert config.chroma_dir == Path("data/shared-chroma")


def test_training_parse_args_uses_shared_defaults(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("TRAIN_BATCH_SIZE", "64")
    monkeypatch.setenv("TRAIN_OUTPUT_DIR", "runs/from-env")
    clear_config_cache()

    args = parse_train_args([])

    assert args.batch_size == 64
    assert args.output_dir == Path("runs/from-env")


def test_demo_app_config_uses_shared_rag_top_k(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RAG_TOP_K", "7")
    clear_config_cache()

    config = DemoAppConfig.from_env()

    assert config.rag_top_k == 7


def test_create_app_uses_shared_config_by_default(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("APP_TITLE", "Configured Demo")
    clear_config_cache()
    runtime = DemoRuntime(
        artifact_bundle=type("Artifacts", (), {"artifact_dir": "tmp"})(),
        inference_engine=type("Inference", (), {})(),
        initial_workflow=type("InitialWorkflow", (), {})(),
        followup_workflow=type("FollowupWorkflow", (), {})(),
        rag_status={"index_ready": True, "chunk_count": 1},
    )

    app = create_app(runtime=runtime)

    assert app.title == "Configured Demo"
