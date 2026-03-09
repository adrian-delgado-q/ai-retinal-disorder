from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Class weights cannot be computed because some classes are absent: {counts}")
    total = counts.sum()
    weights = total / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def assert_no_image_overlap(*frames) -> None:
    seen: dict[str, str] = {}
    for split_name, frame in frames:
        for image_id in frame["image_id"].astype(str):
            previous = seen.get(image_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate image_id '{image_id}' found in both '{previous}' and '{split_name}'."
                )
            seen[image_id] = split_name


@dataclass
class EarlyStopping:
    patience: int
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

