from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo_app.artifacts import sync_model_artifacts, validate_artifact_dir


def _write_artifact_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "best_model.pt").write_bytes(b"checkpoint")
    (root / "label_map.json").write_text(json.dumps({"glaucoma": 0, "healthy": 1}), encoding="utf-8")
    (root / "run_config.json").write_text(json.dumps({"model": "resnet18", "img_size": 224}), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")


def test_validate_artifact_dir_requires_expected_files(tmp_path: Path):
    _write_artifact_tree(tmp_path)
    bundle = validate_artifact_dir(tmp_path)
    assert bundle.metrics == {"accuracy": 0.9}

    (tmp_path / "label_map.json").unlink()
    with pytest.raises(RuntimeError) as exc:
        validate_artifact_dir(tmp_path)
    assert "label_map.json" in str(exc.value)


def test_sync_model_artifacts_copies_local_directory(tmp_path: Path):
    source_dir = tmp_path / "source"
    _write_artifact_tree(source_dir)
    target_dir = tmp_path / "cache"

    bundle = sync_model_artifacts(
        source_url=str(source_dir),
        local_dir=target_dir,
        force_refresh=True,
    )

    assert bundle.artifact_dir == target_dir.resolve()
    assert (target_dir / "best_model.pt").exists()


def test_sync_model_artifacts_uses_existing_cache_when_source_url_missing(tmp_path: Path):
    target_dir = tmp_path / "cache"
    _write_artifact_tree(target_dir)

    bundle = sync_model_artifacts(
        source_url=None,
        local_dir=target_dir,
        force_refresh=False,
    )

    assert bundle.artifact_dir == target_dir.resolve()
    assert bundle.metrics == {"accuracy": 0.9}


def test_sync_model_artifacts_requires_source_url_when_cache_missing(tmp_path: Path):
    target_dir = tmp_path / "cache"

    with pytest.raises(RuntimeError) as exc:
        sync_model_artifacts(
            source_url=None,
            local_dir=target_dir,
            force_refresh=False,
        )

    assert "MODEL_ARTIFACT_SOURCE_URL is required" in str(exc.value)
