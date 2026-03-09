from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from rag._deps import require_module
from runtime_logging import configure_logging

from .artifacts import validate_artifact_dir

configure_logging()
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PredictionEntry:
    label: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str
    predicted_confidence: float
    top_predictions: list[PredictionEntry]
    warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.predicted_label,
            "confidence": self.predicted_confidence,
            "top_predictions": [
                {"label": entry.label, "probability": entry.probability} for entry in self.top_predictions
            ],
            "warning": self.warning,
        }


class RetinalInferenceEngine:
    def __init__(
        self,
        *,
        artifact_dir: Path,
        model,
        class_names: list[str],
        eval_transform,
        device,
        low_confidence_threshold: float,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.model = model
        self.class_names = class_names
        self.eval_transform = eval_transform
        self.device = device
        self.low_confidence_threshold = low_confidence_threshold

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: Path,
        *,
        device: str = "cpu",
        low_confidence_threshold: float = 0.55,
    ) -> "RetinalInferenceEngine":
        torch = _torch_module()
        timm = _timm_module()
        artifact_bundle = validate_artifact_dir(artifact_dir)
        artifact_dir = artifact_bundle.artifact_dir
        label_map = json.loads((artifact_dir / "label_map.json").read_text(encoding="utf-8"))
        run_config = json.loads((artifact_dir / "run_config.json").read_text(encoding="utf-8"))

        class_to_index = {key: int(value) for key, value in label_map.items()}
        index_to_class = {value: key for key, value in class_to_index.items()}
        class_names = [index_to_class[index] for index in sorted(index_to_class)]

        image_size = int(run_config["img_size"])
        eval_transform = _build_eval_transform(image_size)

        resolved_device = torch.device(device)
        logger.info(
            "Loading inference artifacts dir=%s model=%s image_size=%s device=%s classes=%s",
            artifact_dir,
            run_config["model"],
            image_size,
            resolved_device,
            class_names,
        )
        checkpoint = torch.load(artifact_dir / "best_model.pt", map_location=resolved_device)
        model = timm.create_model(run_config["model"], pretrained=False, num_classes=len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)
        model.eval()

        return cls(
            artifact_dir=artifact_dir,
            model=model,
            class_names=class_names,
            eval_transform=eval_transform,
            device=resolved_device,
            low_confidence_threshold=low_confidence_threshold,
        )

    def predict_bytes(self, image_bytes: bytes, *, top_k: int = 3) -> PredictionResult:
        torch = _torch_module()
        logger.info("Running prediction image_bytes=%s top_k=%s", len(image_bytes), top_k)
        image = self._open_image(image_bytes)
        x = self.eval_transform(image)
        if hasattr(x, "unsqueeze"):
            x = x.unsqueeze(0)
        if hasattr(x, "to"):
            x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        top_values, top_indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        top_predictions = [
            PredictionEntry(
                label=self.class_names[int(index)],
                probability=round(float(value), 6),
            )
            for value, index in zip(top_values, top_indices)
        ]
        predicted_label = top_predictions[0].label
        predicted_confidence = top_predictions[0].probability
        warning = None
        if predicted_confidence < self.low_confidence_threshold:
            warning = (
                f"Low-confidence prediction ({predicted_confidence:.3f}). "
                "Review the uploaded image and treat the result as exploratory only."
            )
        logger.info(
            "Prediction result label=%s confidence=%.3f warning=%s",
            predicted_label,
            predicted_confidence,
            bool(warning),
        )

        return PredictionResult(
            predicted_label=predicted_label,
            predicted_confidence=predicted_confidence,
            top_predictions=top_predictions,
            warning=warning,
        )

    @staticmethod
    def _open_image(image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc
        return image


def _torch_module():
    return require_module("torch")


def _timm_module():
    return require_module("timm")


def _build_eval_transform(image_size: int):
    transforms = require_module("torchvision.transforms")
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
