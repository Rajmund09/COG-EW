from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders.radioml_dataset import RadioMLDataset
from perception.models.radioml_cnn import RadioMLCNN
from perception.models.radioml_resnet import RadioMLResNet


def resolve_hdf5_path(custom_path: str | None = None) -> Path:
    """Resolve RadioML HDF5 path with support for star-emoji and glob fallbacks."""
    if custom_path:
        p = Path(custom_path).expanduser()
        if p.is_file():
            return p

    candidates = [
        Path(r"D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"),
        Path(r"D:\Datasets\RadioML 2018.01A Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5"),
        ROOT / "data" / "GOLD_XYZ_OSC.0001_1024.hdf5",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for search_root in [Path(r"D:\Datasets"), ROOT / "data"]:
        if search_root.is_dir():
            try:
                for candidate in search_root.glob("*RadioML*/GOLD_XYZ_OSC.0001_1024.hdf5"):
                    if candidate.is_file():
                        return candidate
                for candidate in search_root.rglob("GOLD_XYZ_OSC.0001_1024.hdf5"):
                    if candidate.is_file():
                        return candidate
            except OSError:
                pass

    return candidates[0]


DEFAULT_HDF5 = str(resolve_hdf5_path())

INDEX_PATH = ROOT / "data" / "processed" / "radioml" / "radioml_index.npz"
SPLIT_DIR = (
    ROOT / "data" / "processed" / "radioml" / "experiments" / "split_60_10_30_v1"
)

EXPERIMENT_DIR = ROOT / "models" / "experiments" / "radioml_60_10_30_v1"
REPORT_DIR = ROOT / "reports" / "experiments" / "radioml_60_10_30_v1"

NUM_CLASSES = 24
SEED = 42

# Candidates retained from the preliminary five-fold pilot.
CANDIDATES = [
    {
        "name": "cnn_lr1e-3_wd1e-4",
        "model": "cnn",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
    },
    {
        "name": "resnet_lr2.5e-4_wd5e-4",
        "model": "resnet",
        "learning_rate": 2.5e-4,
        "weight_decay": 5e-4,
        "batch_size": 32,
    },
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str) -> nn.Module:
    if model_name == "cnn":
        return RadioMLCNN(num_classes=NUM_CLASSES)

    if model_name == "resnet":
        return RadioMLResNet(num_classes=NUM_CLASSES)

    raise ValueError(f"Unsupported model: {model_name}")


def make_loader(
    hdf5_path: Path,
    indices: np.ndarray,
    name: str,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    # These are experiment-specific index files, not the source splits.
    index_path = REPORT_DIR / "loader_indices" / f"{name}.npy"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(index_path, indices.astype(np.int32), allow_pickle=False)

    dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=index_path,
        metadata_path=INDEX_PATH,
        normalize=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def close_loader(loader: DataLoader | None) -> None:
    if loader is not None and hasattr(loader.dataset, "close"):
        loader.dataset.close()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()

    for batch_number, (signals, labels, _) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_count

        if batch_number % 500 == 0:
            print(f"    Processed {batch_number:,}/{len(loader):,} batches")

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    include_predictions: bool = False,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    labels_all = []
    predictions_all = []

    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(signals)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        total_samples += batch_count

        labels_all.extend(labels.cpu().numpy().tolist())
        predictions_all.extend(predictions.cpu().numpy().tolist())

    result = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions_all,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "samples": total_samples,
    }

    if include_predictions:
        result["classification_report"] = classification_report(
            labels_all,
            predictions_all,
            labels=list(range(NUM_CLASSES)),
            output_dict=True,
            zero_division=0,
        )
        result["confusion_matrix"] = confusion_matrix(
            labels_all,
            predictions_all,
            labels=list(range(NUM_CLASSES)),
        ).tolist()

    return result


def choose_subset(
    indices: np.ndarray,
    limit: int,
    seed: int,
) -> np.ndarray:
    if limit <= 0 or limit >= len(indices):
        return indices.copy()

    rng = np.random.default_rng(seed)
    # The saved split indices are already shuffled by the split generator.
    # Use a deterministic random subset for smoke testing.
    chosen = rng.choice(indices, size=limit, replace=False)
    return chosen.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare RadioML CNN and ResNet finalists."
    )
    parser.add_argument("--hdf5", default=DEFAULT_HDF5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)

    # Optional limits are only for pipeline smoke tests.
    # Zero means use the complete partition.
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Smoke-test mode: do not load or evaluate the test partition.",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be at least 1.")

    hdf5_path = resolve_hdf5_path(args.hdf5)

    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"RadioML file not found: {hdf5_path}\n"
            "Pass the correct file using --hdf5."
        )

    for path in [
        INDEX_PATH,
        SPLIT_DIR / "train_indices.npy",
        SPLIT_DIR / "validation_indices.npy",
        SPLIT_DIR / "test_indices.npy",
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_indices = np.load(SPLIT_DIR / "train_indices.npy", allow_pickle=False).astype(
        np.int64
    )

    validation_indices = np.load(
        SPLIT_DIR / "validation_indices.npy", allow_pickle=False
    ).astype(np.int64)

    # Read the test index file only after validation has selected the winner.
    # In smoke-test mode it is not read at all.
    if args.train_limit:
        train_indices = choose_subset(train_indices, args.train_limit, args.seed)

    if args.validation_limit:
        validation_indices = choose_subset(
            validation_indices, args.validation_limit, args.seed + 1
        )

    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Training/validation overlap detected.")

    print("=" * 72)
    print("COG-EW — RADIO ML FINALIST TRAINING")
    print("=" * 72)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"Training samples: {len(train_indices):,}")
    print(f"Validation samples: {len(validation_indices):,}")
    print(f"Epoch limit: {args.epochs}")
    print(f"Test evaluation enabled: {not args.skip_test}")

    if args.train_limit or args.validation_limit:
        print("[SMOKE TEST] One or more partitions are intentionally reduced.")

    if args.skip_test:
        print("[SMOKE TEST] Test indices will not be loaded.")

    model_results = []
    experiment_started = time.perf_counter()

    for candidate in CANDIDATES:
        print("\n" + "=" * 72)
        print(f"Training candidate: {candidate['name']}")
        print("=" * 72)

        train_loader = None
        validation_loader = None
        model = None

        try:
            train_loader = make_loader(
                hdf5_path,
                train_indices,
                f"{candidate['name']}_train",
                candidate["batch_size"],
                True,
                device,
            )
            validation_loader = make_loader(
                hdf5_path,
                validation_indices,
                f"{candidate['name']}_validation",
                candidate["batch_size"],
                False,
                device,
            )

            model = build_model(candidate["model"]).to(device)
            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=candidate["learning_rate"],
                weight_decay=candidate["weight_decay"],
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=1,
                min_lr=1e-6,
            )

            best_macro_f1 = -1.0
            best_epoch = 0
            stale_epochs = 0
            history = []

            checkpoint_path = EXPERIMENT_DIR / f"{candidate['name']}_best.pt"

            for epoch in range(1, args.epochs + 1):
                print(f"\nEpoch {epoch}/{args.epochs}")

                train_metrics = train_epoch(
                    model, train_loader, optimizer, criterion, device
                )

                validation_metrics = evaluate(
                    model, validation_loader, criterion, device
                )

                scheduler.step(validation_metrics["macro_f1"])
                current_lr = optimizer.param_groups[0]["lr"]

                row = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "learning_rate": current_lr,
                }
                history.append(row)

                print(
                    f"Train loss={train_metrics['loss']:.4f}, "
                    f"train acc={train_metrics['accuracy']:.4f} | "
                    f"val loss={validation_metrics['loss']:.4f}, "
                    f"val acc={validation_metrics['accuracy']:.4f}, "
                    f"val macro-F1={validation_metrics['macro_f1']:.4f}"
                )

                if validation_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = validation_metrics["macro_f1"]
                    best_epoch = epoch
                    stale_epochs = 0

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_name": candidate["model"],
                            "candidate": candidate,
                            "best_epoch": best_epoch,
                            "validation_macro_f1": best_macro_f1,
                            "seed": args.seed,
                            "train_samples": len(train_indices),
                            "validation_samples": len(validation_indices),
                        },
                        checkpoint_path,
                    )
                    print("[CHECKPOINT] Best validation macro-F1 saved.")
                else:
                    stale_epochs += 1
                    if stale_epochs >= args.patience:
                        print("[EARLY STOP] Validation macro-F1 did not improve.")
                        break

            with (REPORT_DIR / f"{candidate['name']}_history.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(history, file, indent=2)

            model_results.append(
                {
                    "candidate": candidate,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "checkpoint": str(checkpoint_path),
                    "history": history,
                }
            )

        finally:
            close_loader(train_loader)
            close_loader(validation_loader)
            del model, train_loader, validation_loader
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not model_results:
        raise RuntimeError("No candidate completed training.")

    # Select only on the 10% validation partition.
    model_results.sort(
        key=lambda row: row["best_validation_macro_f1"],
        reverse=True,
    )
    winner = model_results[0]

    print("\n" + "=" * 72)
    print("VALIDATION-BASED MODEL SELECTION")
    print("=" * 72)

    for rank, result in enumerate(model_results, start=1):
        print(
            f"{rank}. {result['candidate']['name']} | "
            f"best val macro-F1="
            f"{result['best_validation_macro_f1']:.4f} | "
            f"epoch={result['best_epoch']}"
        )

    print(f"\nSelected model: {winner['candidate']['name']}")

    report = {
        "experiment": "RadioML 2018.01A 60/10/30 finalists",
        "seed": args.seed,
        "device": str(device),
        "training_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_evaluation_enabled": not args.skip_test,
        "selection_metric": "validation macro-F1",
        "candidates": [
            {
                "name": row["candidate"]["name"],
                "hyperparameters": row["candidate"],
                "best_epoch": row["best_epoch"],
                "best_validation_macro_f1": row["best_validation_macro_f1"],
                "checkpoint": row["checkpoint"],
            }
            for row in model_results
        ],
        "selected_candidate": winner["candidate"]["name"],
        "selected_checkpoint": winner["checkpoint"],
        "final_test_evaluation": None,
        "elapsed_before_test_seconds": (time.perf_counter() - experiment_started),
        "notes": [
            "Finalist configurations came from a preliminary pilot CV.",
            "Model selection uses only the separate validation partition.",
            "The test partition is not used for training or selection.",
        ],
    }

    if not args.skip_test:
        # This is the only section that reads the test indices.
        test_indices = np.load(
            SPLIT_DIR / "test_indices.npy", allow_pickle=False
        ).astype(np.int64)

        if (
            np.intersect1d(train_indices, test_indices).size
            or np.intersect1d(validation_indices, test_indices).size
        ):
            raise RuntimeError("Test partition overlaps another partition.")

        test_loader = None
        selected_model = None

        try:
            selected_model = build_model(winner["candidate"]["model"]).to(device)

            checkpoint = torch.load(
                winner["checkpoint"],
                map_location=device,
                weights_only=True,
            )
            selected_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

            test_loader = make_loader(
                hdf5_path,
                test_indices,
                "selected_model_test",
                winner["candidate"]["batch_size"],
                False,
                device,
            )

            test_metrics = evaluate(
                selected_model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
                include_predictions=True,
            )

            confusion = np.asarray(test_metrics.pop("confusion_matrix"), dtype=np.int64)
            np.savetxt(
                REPORT_DIR / "selected_model_test_confusion_matrix.csv",
                confusion,
                delimiter=",",
                fmt="%d",
            )

            report["final_test_evaluation"] = test_metrics
            report["final_test_evaluation"]["samples"] = len(test_indices)
            report["test_confusion_matrix_csv"] = str(
                REPORT_DIR / "selected_model_test_confusion_matrix.csv"
            )
            report["test_used_for_training"] = False
            report["test_used_for_selection"] = False

            print("\n" + "=" * 72)
            print("FINAL LOCKED TEST RESULT — SELECTED MODEL ONLY")
            print("=" * 72)
            print(f"Test samples: {len(test_indices):,}")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
            print(f"Test loss: {test_metrics['loss']:.4f}")
            print("Per-class report: stored in the JSON experiment report.")

        finally:
            close_loader(test_loader)
            del selected_model, test_loader
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report["elapsed_total_seconds"] = time.perf_counter() - experiment_started

    report_path = REPORT_DIR / "finalist_comparison_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nExperiment report: {report_path}")
    print(f"New checkpoints: {EXPERIMENT_DIR}")
    print("\n[PASS] Finalist experiment completed.")


if __name__ == "__main__":
    main()
