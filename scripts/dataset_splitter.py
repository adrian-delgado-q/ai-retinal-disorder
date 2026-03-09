import argparse
import csv
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_EXCLUDED_CLASSES = {"pterygium"}
DEFAULT_REVIEW_STATUS = "unreviewed"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MANIFEST_COLUMNS = [
    "image_id",
    "file_path",
    "class_name",
    "class_id",
    "source_set",
    "split",
    "review_status",
    "image_quality",
    "modality_check",
    "duplicate_group",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits from the original Mendeley eye dataset."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing 'original/' and optionally 'augmented/' subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Directory where reviewed/, splits/, and optional processed/ outputs will be written.",
    )
    parser.add_argument(
        "--include-class",
        dest="include_classes",
        action="append",
        default=[],
        help="Class folder name to include. Repeat for multiple classes.",
    )
    parser.add_argument(
        "--exclude-class",
        dest="exclude_classes",
        action="append",
        default=[],
        help="Class folder name to exclude. Repeat for multiple classes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic per-class shuffling before splitting.",
    )
    parser.add_argument(
        "--materialize",
        choices=("none", "copy", "symlink"),
        default="none",
        help="How to populate processed/train|val|test folders from the split manifests.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    text = value.strip().lower()
    slug = []
    previous_was_separator = False
    for char in text:
        if char.isalnum():
            slug.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            slug.append("_")
            previous_was_separator = True
    return "".join(slug).strip("_")


def list_class_dirs(original_root: Path) -> list[Path]:
    if not original_root.exists():
        raise FileNotFoundError(f"Original dataset directory not found: {original_root}")
    class_dirs = [path for path in sorted(original_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under: {original_root}")
    return class_dirs


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_manifest_rows(
    original_root: Path,
    include_classes: set[str],
    exclude_classes: set[str],
) -> tuple[list[dict[str, str]], dict[str, int], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    class_counts: dict[str, int] = {}
    audit_rows: list[dict[str, str]] = []
    class_dirs = list_class_dirs(original_root)

    active_class_dirs = []
    for class_dir in class_dirs:
        folder_name = class_dir.name
        normalized = slugify(folder_name)
        image_paths = collect_images(class_dir)
        audit_rows.append(
            {
                "folder_name": folder_name,
                "normalized_name": normalized,
                "included": "",
                "image_count": str(len(image_paths)),
            }
        )

        if include_classes and normalized not in include_classes:
            continue
        if normalized in exclude_classes:
            continue
        if not image_paths:
            continue
        active_class_dirs.append((folder_name, normalized, image_paths))
        class_counts[normalized] = len(image_paths)

    if not active_class_dirs:
        raise ValueError("No eligible class folders found after include/exclude filtering.")

    class_id_map = {
        normalized: class_id
        for class_id, normalized in enumerate(sorted(class_counts))
    }

    image_counter = 1
    for folder_name, normalized, image_paths in active_class_dirs:
        for image_path in image_paths:
            rows.append(
                {
                    "image_id": f"{image_counter:06d}",
                    "file_path": str(image_path),
                    "class_name": normalized,
                    "class_id": str(class_id_map[normalized]),
                    "source_set": "original",
                    "split": "",
                    "review_status": DEFAULT_REVIEW_STATUS,
                    "image_quality": "",
                    "modality_check": "",
                    "duplicate_group": "",
                    "notes": "",
                }
            )
            image_counter += 1

    for audit_row in audit_rows:
        normalized = audit_row["normalized_name"]
        audit_row["included"] = "yes" if normalized in class_counts else "no"

    return rows, class_counts, audit_rows


def split_counts_for_class(class_count: int) -> dict[str, int]:
    if class_count < 3:
        raise ValueError(
            f"Class with {class_count} images is too small for a 70/15/15 split. Minimum is 3."
        )

    exact = {split: class_count * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: math.floor(value) for split, value in exact.items()}
    remainder = class_count - sum(counts.values())

    for split, _ in sorted(
        SPLIT_RATIOS.items(),
        key=lambda item: (exact[item[0]] - counts[item[0]], item[0]),
        reverse=True,
    ):
        if remainder == 0:
            break
        counts[split] += 1
        remainder -= 1

    if any(count == 0 for count in counts.values()):
        raise ValueError(
            f"Class with {class_count} images cannot provide at least one sample to each split."
        )

    return counts


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_class[row["class_name"]].append(row)

    assigned_rows: list[dict[str, str]] = []
    for class_name in sorted(rows_by_class):
        class_rows = rows_by_class[class_name]
        counts = split_counts_for_class(len(class_rows))

        def ordering_key(row: dict[str, str]) -> tuple[int, str]:
            relative_path = Path(row["file_path"]).name
            salted = f"{seed}:{class_name}:{relative_path}:{row['image_id']}"
            return (stable_hash(salted), relative_path)

        ordered_rows = sorted(class_rows, key=ordering_key)

        start = 0
        for split in ("train", "val", "test"):
            end = start + counts[split]
            for row in ordered_rows[start:end]:
                row["split"] = split
                assigned_rows.append(row)
            start = end

    return sorted(assigned_rows, key=lambda row: int(row["image_id"]))


def stable_hash(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % (2**63 - 1)
    return value


def ensure_directories(output_root: Path) -> tuple[Path, Path, Path]:
    reviewed_dir = output_root / "reviewed"
    splits_dir = output_root / "splits"
    processed_dir = output_root / "processed"
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return reviewed_dir, splits_dir, processed_dir


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def materialize_splits(rows: list[dict[str, str]], processed_dir: Path, mode: str) -> None:
    if mode == "none":
        return

    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        source = Path(row["file_path"])
        target_dir = processed_dir / row["split"] / row["class_name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name

        if target.exists() or target.is_symlink():
            target.unlink()

        if mode == "copy":
            shutil.copy2(source, target)
        else:
            target.symlink_to(source.resolve())


def build_summary_rows(rows: list[dict[str, str]], audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    split_counts = Counter(row["split"] for row in rows)
    class_split_counts: dict[tuple[str, str], int] = Counter(
        (row["class_name"], row["split"]) for row in rows
    )

    summary_rows: list[dict[str, str]] = []
    for audit_row in audit_rows:
        class_name = audit_row["normalized_name"]
        if audit_row["included"] != "yes":
            summary_rows.append(
                {
                    "class_name": class_name,
                    "included": "no",
                    "total_images": audit_row["image_count"],
                    "train_images": "0",
                    "val_images": "0",
                    "test_images": "0",
                }
            )
            continue

        summary_rows.append(
            {
                "class_name": class_name,
                "included": "yes",
                "total_images": audit_row["image_count"],
                "train_images": str(class_split_counts[(class_name, "train")]),
                "val_images": str(class_split_counts[(class_name, "val")]),
                "test_images": str(class_split_counts[(class_name, "test")]),
            }
        )

    summary_rows.append(
        {
            "class_name": "__totals__",
            "included": "yes",
            "total_images": str(len(rows)),
            "train_images": str(split_counts["train"]),
            "val_images": str(split_counts["val"]),
            "test_images": str(split_counts["test"]),
        }
    )
    return summary_rows


def main() -> None:
    args = parse_args()

    raw_root = args.raw_root
    original_root = raw_root / "original"
    _augmented_root = raw_root / "augmented"

    include_classes = {slugify(value) for value in args.include_classes}
    exclude_classes = DEFAULT_EXCLUDED_CLASSES | {slugify(value) for value in args.exclude_classes}

    rows, class_counts, audit_rows = build_manifest_rows(
        original_root=original_root,
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )
    rows = assign_splits(rows, seed=args.seed)

    reviewed_dir, splits_dir, processed_dir = ensure_directories(args.output_root)

    write_csv(reviewed_dir / "labels_master.csv", rows, MANIFEST_COLUMNS)
    write_csv(splits_dir / "train.csv", build_split_rows(rows, "train"), MANIFEST_COLUMNS)
    write_csv(splits_dir / "val.csv", build_split_rows(rows, "val"), MANIFEST_COLUMNS)
    write_csv(splits_dir / "test.csv", build_split_rows(rows, "test"), MANIFEST_COLUMNS)

    summary_rows = build_summary_rows(rows, audit_rows)
    write_csv(
        reviewed_dir / "dataset_summary.csv",
        summary_rows,
        ["class_name", "included", "total_images", "train_images", "val_images", "test_images"],
    )

    materialize_splits(rows, processed_dir, mode=args.materialize)

    print(f"Original classes included: {len(class_counts)}")
    print(f"Original images included: {len(rows)}")
    print(f"Excluded classes by default: {', '.join(sorted(DEFAULT_EXCLUDED_CLASSES))}")
    print("Split totals:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {sum(1 for row in rows if row['split'] == split)}")
    print(f"Master manifest: {reviewed_dir / 'labels_master.csv'}")
    print(f"Summary report: {reviewed_dir / 'dataset_summary.csv'}")
    print(f"Split manifests: {splits_dir}")
    if args.materialize != "none":
        print(f"Processed folders: {processed_dir}")


if __name__ == "__main__":
    main()
