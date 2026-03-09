from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import SUPPORTED_MODELS, TOP5_CLASSES, get_training_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = get_training_settings()
    parser = argparse.ArgumentParser(description="Train a retinal disease classifier from CSV manifests.")
    parser.add_argument("--train-csv", type=Path, default=defaults.train_csv)
    parser.add_argument("--val-csv", type=Path, default=defaults.val_csv)
    parser.add_argument("--test-csv", type=Path, default=defaults.test_csv)
    parser.add_argument("--data-root", type=Path, default=defaults.data_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default=defaults.model_name)
    parser.add_argument("--img-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--allow-provisional-data", action="store_true", default=defaults.allow_provisional_data)
    parser.add_argument("--run-test-after-training", action="store_true", default=defaults.run_test_after_training)
    return parser.parse_args(argv)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace):
    from torch.utils.data import DataLoader

    from training.dataset import (
        RetinalCSVDataset,
        load_dataset_bundle,
        load_split_frame,
        resolve_csv_paths,
    )
    from training.utils import assert_no_image_overlap

    defaults = get_training_settings()
    train_csv, val_csv, test_csv, data_mode = resolve_csv_paths(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        provisional_train_csv=defaults.provisional_train_csv,
        provisional_val_csv=defaults.provisional_val_csv,
        provisional_test_csv=defaults.provisional_test_csv,
        allow_provisional_data=args.allow_provisional_data,
    )

    bundle = load_dataset_bundle(
        train_csv=train_csv,
        classes=TOP5_CLASSES,
        data_root=args.data_root,
        data_mode=data_mode,
        max_samples=args.max_train_samples,
    )
    train_frame = bundle.frame
    val_frame = load_split_frame(
        csv_path=val_csv,
        split_name="val",
        classes=tuple(bundle.class_to_index.keys()),
        data_root=args.data_root,
        data_mode=data_mode,
        max_samples=args.max_val_samples,
    )
    test_frame = load_split_frame(
        csv_path=test_csv,
        split_name="test",
        classes=tuple(bundle.class_to_index.keys()),
        data_root=args.data_root,
        data_mode=data_mode,
        max_samples=args.max_test_samples,
    )

    assert_no_image_overlap(("train", train_frame), ("val", val_frame), ("test", test_frame))

    train_transform, eval_transform = build_transforms(args.img_size)
    train_dataset = RetinalCSVDataset(train_frame, bundle.class_to_index, transform=train_transform)
    val_dataset = RetinalCSVDataset(val_frame, bundle.class_to_index, transform=eval_transform)
    test_dataset = RetinalCSVDataset(test_frame, bundle.class_to_index, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_frame, val_frame, test_frame, bundle.class_to_index, data_mode, (train_csv, val_csv, test_csv)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    from torch.cuda.amp import autocast

    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def evaluate_model(model, loader, criterion, device, class_names):
    import torch
    from torch.cuda.amp import autocast

    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    from training.evaluate import compute_metrics

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    metrics = compute_metrics(labels_all, predictions_all, class_names)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics, labels_all, predictions_all


def prepare_output_dir(output_dir: Path, data_mode: str) -> Path:
    if data_mode == "provisional" and "provisional" not in output_dir.name:
        output_dir = output_dir.parent / f"{output_dir.name}_provisional"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()

    import pandas as pd
    import timm
    import torch
    from torch import nn
    from torch.cuda.amp import GradScaler

    from training.evaluate import save_confusion_matrix
    from training.utils import (
        EarlyStopping,
        compute_class_weights,
        save_checkpoint,
        save_history_csv,
        save_json,
        set_seed,
    )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_frame, val_frame, test_frame, class_to_index, data_mode, resolved_csvs = build_loaders(args)
    output_dir = prepare_output_dir(args.output_dir, data_mode)

    if data_mode == "provisional":
        print("WARNING: running in provisional mode with unreviewed split manifests.")

    class_names = [class_name for class_name, _ in sorted(class_to_index.items(), key=lambda item: item[1])]
    train_labels = [class_to_index[value] for value in train_frame["class_name"].tolist()]
    class_weights = compute_class_weights(train_labels, num_classes=len(class_names)).to(device)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    early_stopping = EarlyStopping(patience=args.patience)

    run_config = {
        "train_csv": str(resolved_csvs[0]),
        "val_csv": str(resolved_csvs[1]),
        "test_csv": str(resolved_csvs[2]),
        "data_root": str(args.data_root) if args.data_root else None,
        "data_mode": data_mode,
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "seed": args.seed,
        "classes": class_names,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "device": str(device),
        "provisional": data_mode == "provisional",
    }
    save_json(output_dir / "run_config.json", run_config)
    save_json(output_dir / "label_map.json", {name: index for name, index in class_to_index.items()})

    best_state = None
    best_val_metrics = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics, val_labels, val_predictions = evaluate_model(
            model, val_loader, criterion, device, class_names
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_weighted_f1": round(val_metrics["weighted_f1"], 6),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )

        if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_val_metrics = val_metrics
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_index": class_to_index,
                "run_config": run_config,
            }
            save_checkpoint(output_dir / "best_model.pt", best_state)
            save_confusion_matrix(
                val_labels,
                val_predictions,
                class_names,
                output_dir / "val_confusion_matrix.png",
                title="Validation Confusion Matrix",
            )

        save_checkpoint(
            output_dir / "last_model.pt",
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_index": class_to_index,
                "run_config": run_config,
            },
        )

        if early_stopping.step(val_metrics["macro_f1"]):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    save_history_csv(output_dir / "history.csv", history)

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])

    metrics_payload = {
        "data_mode": data_mode,
        "provisional": data_mode == "provisional",
        "class_names": class_names,
        "best_epoch": best_state["epoch"],
        "best_validation": best_val_metrics,
    }

    if args.run_test_after_training:
        test_metrics, test_labels, test_predictions = evaluate_model(
            model, test_loader, criterion, device, class_names
        )
        metrics_payload["test"] = test_metrics
        save_confusion_matrix(
            test_labels,
            test_predictions,
            class_names,
            output_dir / "test_confusion_matrix.png",
            title="Test Confusion Matrix",
        )

    save_json(output_dir / "metrics.json", metrics_payload)

    counts = pd.Series(train_frame["class_name"]).value_counts().to_dict()
    print("Training complete.")
    print(f"Data mode: {data_mode}")
    print(f"Output dir: {output_dir}")
    print(f"Train class counts: {counts}")


if __name__ == "__main__":
    main()
