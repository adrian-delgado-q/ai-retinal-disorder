from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from training.config import (
    APPROVED_IMAGE_QUALITIES,
    APPROVED_MODALITY,
    APPROVED_REVIEW_STATUS,
    REQUIRED_COLUMNS,
)


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    class_to_index: dict[str, int]
    resolved_data_mode: str


def resolve_csv_paths(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    provisional_train_csv: Path,
    provisional_val_csv: Path,
    provisional_test_csv: Path,
    allow_provisional_data: bool,
) -> tuple[Path, Path, Path, str]:
    explicit_paths = (train_csv, val_csv, test_csv)
    if all(path.exists() for path in explicit_paths):
        explicit_mode = infer_manifest_mode(train_csv, val_csv, test_csv)
        if explicit_mode == "provisional" and not allow_provisional_data:
            raise FileNotFoundError(
                "Provisional manifests were provided explicitly, but "
                "--allow-provisional-data was not set."
            )
        return train_csv, val_csv, test_csv, explicit_mode

    clean_paths = (train_csv, val_csv, test_csv)
    if all(path.exists() for path in clean_paths):
        return train_csv, val_csv, test_csv, "clean"

    if not allow_provisional_data:
        missing = [str(path) for path in clean_paths if not path.exists()]
        raise FileNotFoundError(
            "Clean manifests are required by default. Missing: " + ", ".join(missing)
        )

    provisional_paths = (provisional_train_csv, provisional_val_csv, provisional_test_csv)
    missing = [str(path) for path in provisional_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Provisional manifests were requested but are missing: " + ", ".join(missing)
        )

    return provisional_train_csv, provisional_val_csv, provisional_test_csv, "provisional"


def infer_manifest_mode(train_csv: Path, val_csv: Path, test_csv: Path) -> str:
    names = {train_csv.name, val_csv.name, test_csv.name}
    if any(name.endswith("_clean.csv") for name in names):
        return "clean"
    return "provisional"


def load_split_frame(
    csv_path: Path,
    split_name: str,
    classes: tuple[str, ...],
    data_root: Path | None,
    data_mode: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    validate_columns(frame, csv_path)

    filtered = frame[frame["class_name"].isin(classes)].copy()

    if data_mode == "clean":
        filtered = filtered[
            (filtered["review_status"] == APPROVED_REVIEW_STATUS)
            & (filtered["modality_check"] == APPROVED_MODALITY)
            & (filtered["image_quality"].isin(APPROVED_IMAGE_QUALITIES))
        ].copy()

    filtered["resolved_path"] = filtered["file_path"].map(
        lambda value: str(resolve_image_path(value, data_root))
    )

    if filtered.empty:
        raise ValueError(f"No usable rows remain in {csv_path} for split '{split_name}'.")

    if max_samples is not None:
        filtered = filtered.head(max_samples).copy()

    return filtered.reset_index(drop=True)


def load_dataset_bundle(
    train_csv: Path,
    classes: tuple[str, ...],
    data_root: Path | None,
    data_mode: str,
    max_samples: int | None = None,
) -> DatasetBundle:
    frame = load_split_frame(
        csv_path=train_csv,
        split_name="train",
        classes=classes,
        data_root=data_root,
        data_mode=data_mode,
        max_samples=max_samples,
    )
    class_names = sorted(frame["class_name"].unique().tolist())
    unknown = set(class_names) - set(classes)
    if unknown:
        raise ValueError(f"Unknown classes found after filtering: {sorted(unknown)}")
    missing = set(classes) - set(class_names)
    if missing:
        raise ValueError(
            "Training data is missing required classes after filtering: "
            + ", ".join(sorted(missing))
        )

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    return DatasetBundle(frame=frame, class_to_index=class_to_index, resolved_data_mode=data_mode)


def validate_columns(frame: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")


def resolve_image_path(raw_path: str, data_root: Path | None) -> Path:
    original = Path(raw_path)
    candidates: list[Path] = []

    if original.is_absolute():
        candidates.append(original)
    else:
        candidates.append(original)
        if data_root is not None:
            candidates.append(data_root / original)
            if original.parts and original.parts[0] == "data":
                candidates.append(data_root / Path(*original.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve image path '{raw_path}' with data_root={data_root}")


class RetinalCSVDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_index: dict[str, int],
        transform=None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["resolved_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_index[row["class_name"]]
        return image, label
