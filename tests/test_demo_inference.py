from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from demo_app.inference import RetinalInferenceEngine


class FakeTensor(list):
    def unsqueeze(self, _dim):
        return self

    def to(self, _device):
        return self

    def cpu(self):
        return self


class FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.loaded_state = None
        self.device = None
        self.eval_called = False

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, _tensor):
        return self.logits


class FakeTorchModule:
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def device(self, name):
        return name

    def load(self, *_args, **_kwargs):
        return {"model_state_dict": {"w": 1}}

    def no_grad(self):
        return self._NoGrad()

    def softmax(self, logits, dim=1):
        row = logits[0]
        exps = [math.exp(value) for value in row]
        total = sum(exps)
        return [FakeTensor([value / total for value in exps])]

    def topk(self, probabilities, k):
        indexed = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)[:k]
        values = FakeTensor([value for _, value in indexed])
        indices = FakeTensor([index for index, _ in indexed])
        return values, indices


def _artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "best_model.pt").write_bytes(b"checkpoint")
    (artifact_dir / "label_map.json").write_text(
        json.dumps({"glaucoma": 0, "healthy": 1, "myopia": 2}),
        encoding="utf-8",
    )
    (artifact_dir / "run_config.json").write_text(
        json.dumps({"model": "resnet18", "img_size": 224}),
        encoding="utf-8",
    )
    return artifact_dir


def _image_bytes() -> bytes:
    image = Image.new("RGB", (256, 256), color=(80, 120, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_inference_engine_loads_artifacts_and_class_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_model = FakeModel([[0.5, 0.2, 0.3]])
    monkeypatch.setattr("demo_app.inference._torch_module", lambda: FakeTorchModule())
    monkeypatch.setattr("demo_app.inference._timm_module", lambda: type("FakeTimm", (), {"create_model": lambda *args, **kwargs: fake_model})())
    monkeypatch.setattr("demo_app.inference._build_eval_transform", lambda image_size: lambda image: FakeTensor([0.0]))

    engine = RetinalInferenceEngine.from_artifacts(_artifact_dir(tmp_path), device="cpu")

    assert engine.class_names == ["glaucoma", "healthy", "myopia"]
    assert fake_model.loaded_state == {"w": 1}
    assert fake_model.eval_called is True


def test_predict_bytes_returns_sorted_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_model = FakeModel([[1.0, 3.0, 2.0]])
    monkeypatch.setattr("demo_app.inference._torch_module", lambda: FakeTorchModule())
    monkeypatch.setattr("demo_app.inference._timm_module", lambda: type("FakeTimm", (), {"create_model": lambda *args, **kwargs: fake_model})())
    monkeypatch.setattr("demo_app.inference._build_eval_transform", lambda image_size: lambda image: FakeTensor([0.0]))
    engine = RetinalInferenceEngine.from_artifacts(
        _artifact_dir(tmp_path),
        device="cpu",
        low_confidence_threshold=0.99,
    )

    result = engine.predict_bytes(_image_bytes(), top_k=3)

    assert result.predicted_label == "healthy"
    assert [entry.label for entry in result.top_predictions] == ["healthy", "myopia", "glaucoma"]
    assert result.warning is not None
