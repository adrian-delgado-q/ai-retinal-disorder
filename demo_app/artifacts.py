from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from rag._deps import require_module
from runtime_logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


REQUIRED_ARTIFACT_FILES = ("best_model.pt", "label_map.json", "run_config.json")
OPTIONAL_ARTIFACT_FILES = ("metrics.json",)


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_dir: Path
    metrics: dict | None


def validate_artifact_dir(artifact_dir: Path) -> ArtifactBundle:
    logger.info("Validating artifact directory %s", artifact_dir)
    missing = [name for name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"Model artifact directory '{artifact_dir}' is missing required files: {', '.join(missing)}"
        )
    metrics_path = artifact_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    return ArtifactBundle(artifact_dir=artifact_dir, metrics=metrics)


def sync_model_artifacts(
    *,
    source_url: str | None,
    local_dir: Path,
    force_refresh: bool = False,
    run_subdir: str | None = None,
) -> ArtifactBundle:
    local_dir = local_dir.resolve()
    logger.info(
        "Syncing model artifacts source_url=%s local_dir=%s force_refresh=%s run_subdir=%s",
        source_url,
        local_dir,
        force_refresh,
        run_subdir,
    )
    if local_dir.exists() and not force_refresh:
        try:
            logger.info("Attempting to reuse local artifact cache at %s", local_dir)
            return validate_artifact_dir(_resolve_candidate_dir(local_dir, run_subdir))
        except RuntimeError:
            logger.warning("Existing local artifact cache is invalid, refreshing", exc_info=True)
            pass

    if not source_url:
        raise RuntimeError(
            "MODEL_ARTIFACT_SOURCE_URL is required when no valid local model artifact cache is available."
        )

    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(source_url)
    if parsed.scheme in {"", "file"}:
        source_path = Path(parsed.path if parsed.scheme == "file" else source_url).expanduser().resolve()
        if not source_path.exists():
            raise RuntimeError(f"Artifact source path does not exist: {source_path}")
        logger.info("Copying local artifact source from %s", source_path)
        _copy_source_directory(source_path, local_dir)
    elif "drive.google.com" in parsed.netloc:
        logger.info("Downloading model artifacts from Google Drive")
        _download_google_drive_folder(source_url, local_dir)
    else:
        raise RuntimeError(
            "Unsupported MODEL_ARTIFACT_SOURCE_URL. Use a local path, file:// path, or a public Google Drive folder URL."
        )

    return validate_artifact_dir(_resolve_candidate_dir(local_dir, run_subdir))


def _copy_source_directory(source_dir: Path, target_dir: Path) -> None:
    if source_dir.is_file():
        raise RuntimeError(f"Artifact source must be a directory, got file: {source_dir}")
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def _download_google_drive_folder(source_url: str, local_dir: Path) -> None:
    try:
        gdown = require_module("gdown")
    except RuntimeError as exc:
        raise RuntimeError(
            "Downloading Google Drive folders requires the optional `gdown` dependency. "
            "Install it via `pip install -r requirements.txt`."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="model-artifacts-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        result = gdown.download_folder(url=source_url, output=str(tmp_path), quiet=True, remaining_ok=True)
        if not result:
            raise RuntimeError(f"Failed to download artifacts from Google Drive folder: {source_url}")
        _copy_source_directory(tmp_path, local_dir)


def _resolve_candidate_dir(root_dir: Path, run_subdir: str | None) -> Path:
    if run_subdir:
        candidate = (root_dir / run_subdir).resolve()
        if not candidate.exists():
            raise RuntimeError(f"Configured MODEL_ARTIFACT_RUN_SUBDIR was not found: {candidate}")
        logger.info("Using configured artifact run subdir %s", candidate)
        return candidate

    if _contains_required_files(root_dir):
        logger.info("Artifact root already contains required files: %s", root_dir)
        return root_dir

    candidates = sorted({path.parent for path in root_dir.rglob("best_model.pt") if _contains_required_files(path.parent)})
    if not candidates:
        raise RuntimeError(
            f"No downloaded artifact directory under '{root_dir}' contains {', '.join(REQUIRED_ARTIFACT_FILES)}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple artifact directories were downloaded. Set MODEL_ARTIFACT_RUN_SUBDIR to choose one explicitly."
        )
    logger.info("Using discovered artifact directory %s", candidates[0])
    return candidates[0]


def _contains_required_files(candidate: Path) -> bool:
    return all((candidate / name).exists() for name in REQUIRED_ARTIFACT_FILES)
