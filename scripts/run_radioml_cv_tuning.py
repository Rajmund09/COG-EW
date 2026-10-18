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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders.radioml_dataset import RadioMLDataset
from perception.models.radioml_cnn import RadioMLCNN
from perception.models.radioml_resnet import RadioMLResNet


INDEX_PATH = ROOT / "data" / "processed" / "radioml" / "radioml_index.npz"

SPLIT_DIR = (
    ROOT / "data" / "processed" / "radioml" / "experiments" / "split_60_10_30_v1"
)

OUTPUT_DIR = ROOT / "reports" / "experiments" / "radioml_cv_tuning_v1"
FOLD_DIR = OUTPUT_DIR / "fold_indices"

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

SEED = 42
NUM_CLASSES = 24
NUM_FOLDS = 5
EPOCHS = 2

# A small pilot search; increase only after timing this run.
SAMPLES_PER_CLASS_SNR = 40

CONFIGS = [
    {
        "model": "cnn",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
    },
    {
        "model": "cnn",
        "learning_rate": 5e-4,
        "weight_decay": 5e-4,
        "batch_size": 64,
    },
    {
        "model": "resnet",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
    },
    {
        "model": "resnet",
        "learning_rate": 2.5e-4,
        "weight_decay": 5e-4,
        "batch_size": 32,
    },
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return RadioMLCNN(num_classes=NUM_CLASSES)

    if name == "resnet":
        return RadioMLResNet(num_classes=NUM_CLASSES)

    raise ValueError(f"Unknown model: {name}")


def load_metadata():
    with np.load(INDEX_PATH, allow_pickle=False) as metadata:
        sample_index = metadata["sample_index"].astype(np.int64)
        class_ids = metadata["class_id"].astype(np.int64)
        snr = metadata["snr"].astype(np.int64)

    if not (len(sample_index) == len(class_ids) == len(snr)):
        raise ValueError("Metadata array lengths do not match.")

    # The existing index builder assigns each HDF5 row its row number.
    if not np.array_equal(sample_index, np.arange(len(sample_index), dtype=np.int64)):
        raise ValueError(
            "Sample IDs are not sequential HDF5 row indices. "
            "Add an explicit sample-ID-to-row mapping before training."
        )

    if not np.array_equal(np.unique(class_ids), np.arange(NUM_CLASSES)):
        raise ValueError("Expected all 24 RadioML class IDs.")

    return sample_index, class_ids, snr


def create_pilot_subset(
    train_ids: np.ndarray,
    class_ids: np.ndarray,
    snr: np.ndarray,
    samples_per_class_snr: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample only from the 60% training partition.
    Stratify by the joint modulation-class/SNR combination.
    """
    rng = np.random.default_rng(seed)

    train_classes = class_ids[train_ids]
    train_snr = snr[train_ids]

    pairs = np.column_stack((train_classes, train_snr))
    unique_pairs = np.unique(pairs, axis=0)

    chosen = []

    for class_id, snr_value in unique_pairs:
        candidates = train_ids[(train_classes == class_id) & (train_snr == snr_value)]

        count = min(samples_per_class_snr, len(candidates))

        if count < NUM_FOLDS:
            raise ValueError(
                f"Only {count} samples available for "
                f"class={class_id}, SNR={snr_value}; "
                "not enough for five-fold CV."
            )

        chosen.extend(rng.choice(candidates, size=count, replace=False).tolist())

    subset_ids = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(subset_ids)

    subset_pairs = np.column_stack((class_ids[subset_ids], snr[subset_ids]))
    _, joint_strata = np.unique(subset_pairs, axis=0, return_inverse=True)

    counts = np.bincount(joint_strata)
    if counts.min() < NUM_FOLDS:
        raise ValueError("A class/SNR stratum has fewer than five samples.")

    return subset_ids, joint_strata


def make_loader(
    hdf5_path: Path,
    ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    index_file: Path,
    device: torch.device,
) -> DataLoader:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_file, ids.astype(np.int32), allow_pickle=False)

    dataset = RadioMLDataset(
        hdf5_path=hdf5_path,
        indices_path=index_file,
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


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for signals, labels, _ in loader:
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
        total += batch_count

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []

    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(signals)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_count = labels.size(0)
        total_loss += loss.item() * batch_count
        total += batch_count

        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "samples": total,
    }


def close_loader(loader: DataLoader | None) -> None:
    if loader is None:
        return

    dataset = loader.dataset
    if hasattr(dataset, "close"):
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot five-fold CNN/ResNet hyperparameter comparison."
    )
    parser.add_argument("--hdf5", default=DEFAULT_HDF5)
    parser.add_argument(
        "--samples-per-class-snr",
        type=int,
        default=SAMPLES_PER_CLASS_SNR,
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.samples_per_class_snr < NUM_FOLDS:
        parser.error("--samples-per-class-snr must be at least 5.")

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    hdf5_path = resolve_hdf5_path(args.hdf5)

    if not hdf5_path.is_file():
        raise FileNotFoundError(
            f"RadioML HDF5 file not found:\n{hdf5_path}\n"
            "Supply the correct path with --hdf5."
        )

    required_files = [
        INDEX_PATH,
        SPLIT_DIR / "train_indices.npy",
        SPLIT_DIR / "validation_indices.npy",
        SPLIT_DIR / "test_indices.npy",
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment file missing: {path}")

    # Verify the split artifacts exist, but do not load validation or test.
    # The test indices are intentionally not read or evaluated in this script.
    train_ids = np.load(SPLIT_DIR / "train_indices.npy", allow_pickle=False).astype(
        np.int64
    )

    _, class_ids, snr = load_metadata()

    if (
        len(np.unique(train_ids)) != len(train_ids)
        or np.any(train_ids < 0)
        or np.any(train_ids >= len(class_ids))
    ):
        raise ValueError("Training split has invalid or duplicate sample IDs.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("RADIOML FIVE-FOLD PILOT MODEL COMPARISON")
    print("=" * 72)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"Training partition: {len(train_ids):,} samples")
    print(f"Pilot samples per class/SNR: {args.samples_per_class_snr}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Fold count: {NUM_FOLDS}")
    print(f"Model configurations: {len(CONFIGS)}")
    print("The separate 10% validation and 30% test data are not evaluated.")

    subset_ids, joint_strata = create_pilot_subset(
        train_ids=train_ids,
        class_ids=class_ids,
        snr=snr,
        samples_per_class_snr=args.samples_per_class_snr,
        seed=args.seed,
    )

    print(f"\nPilot subset: {len(subset_ids):,} samples")

    cv = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=args.seed,
    )

    results = []
    overall_start = time.perf_counter()

    for config_number, config in enumerate(CONFIGS, start=1):
        config_name = (
            f"{config['model']}_"
            f"lr{config['learning_rate']}_"
            f"wd{config['weight_decay']}"
        )

        print("\n" + "=" * 72)
        print(f"CONFIG {config_number}/{len(CONFIGS)}: {config_name}")
        print("=" * 72)

        fold_results = []

        for fold_number, (train_pos, val_pos) in enumerate(
            cv.split(subset_ids, joint_strata),
            start=1,
        ):
            print(f"\nFold {fold_number}/{NUM_FOLDS}")

            fold_train_ids = subset_ids[train_pos]
            fold_val_ids = subset_ids[val_pos]

            if np.intersect1d(fold_train_ids, fold_val_ids).size:
                raise RuntimeError("Fold train/validation overlap detected.")

            train_loader = None
            val_loader = None
            model = None

            try:
                train_loader = make_loader(
                    hdf5_path,
                    fold_train_ids,
                    config["batch_size"],
                    True,
                    FOLD_DIR / f"fold_{fold_number}_train.npy",
                    device,
                )

                val_loader = make_loader(
                    hdf5_path,
                    fold_val_ids,
                    config["batch_size"],
                    False,
                    FOLD_DIR / f"fold_{fold_number}_validation.npy",
                    device,
                )

                model = build_model(config["model"]).to(device)
                criterion = nn.CrossEntropyLoss()

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                )

                fold_start = time.perf_counter()
                last_metrics = None

                for epoch in range(1, args.epochs + 1):
                    train_loss = run_training_epoch(
                        model, train_loader, optimizer, criterion, device
                    )

                    last_metrics = evaluate(model, val_loader, criterion, device)

                    print(
                        f"  Epoch {epoch}/{args.epochs} | "
                        f"train_loss={train_loss:.4f} | "
                        f"fold_val_loss={last_metrics['loss']:.4f} | "
                        f"accuracy={last_metrics['accuracy']:.4f} | "
                        f"macro_f1={last_metrics['macro_f1']:.4f}"
                    )

                fold_result = {
                    "config": config_name,
                    "fold": fold_number,
                    "train_samples": len(fold_train_ids),
                    "validation_samples": len(fold_val_ids),
                    "final_epoch_metrics": last_metrics,
                    "elapsed_seconds": time.perf_counter() - fold_start,
                }

                fold_results.append(fold_result)

            finally:
                close_loader(train_loader)
                close_loader(val_loader)

                del model, train_loader, val_loader
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        accuracy_values = [
            row["final_epoch_metrics"]["accuracy"] for row in fold_results
        ]
        f1_values = [row["final_epoch_metrics"]["macro_f1"] for row in fold_results]

        summary = {
            "config": config_name,
            "hyperparameters": config,
            "folds_completed": len(fold_results),
            "accuracy_mean": float(np.mean(accuracy_values)),
            "accuracy_std": float(np.std(accuracy_values, ddof=1)),
            "macro_f1_mean": float(np.mean(f1_values)),
            "macro_f1_std": float(np.std(f1_values, ddof=1)),
            "fold_results": fold_results,
        }

        results.append(summary)

        print(
            f"\nCV summary: accuracy={summary['accuracy_mean']:.4f}"
            f" ± {summary['accuracy_std']:.4f}; "
            f"macro-F1={summary['macro_f1_mean']:.4f}"
            f" ± {summary['macro_f1_std']:.4f}"
        )

        report_path = OUTPUT_DIR / "cv_tuning_results.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "experiment": "RadioML pilot 5-fold CV",
                    "seed": args.seed,
                    "epochs_per_fold": args.epochs,
                    "samples_per_class_snr": args.samples_per_class_snr,
                    "pilot_samples": len(subset_ids),
                    "folds": NUM_FOLDS,
                    "device": str(device),
                    "configs_completed": len(results),
                    "results": results,
                    "selection_metric": "mean cross-validation macro-F1",
                    "test_partition_used": False,
                    "separate_validation_partition_used": False,
                    "note": (
                        "Pilot subset results are for model selection only. "
                        "They are not final full-data performance estimates. "
                        "The 30% test set must remain untouched until final "
                        "training and evaluation."
                    ),
                    "elapsed_seconds": time.perf_counter() - overall_start,
                },
                file,
                indent=2,
            )

    results.sort(key=lambda row: row["macro_f1_mean"], reverse=True)

    print("\n" + "=" * 72)
    print("PILOT CV RANKING — MACRO-F1")
    print("=" * 72)

    for rank, row in enumerate(results, start=1):
        print(
            f"{rank}. {row['config']} | "
            f"macro-F1={row['macro_f1_mean']:.4f} "
            f"+/- {row['macro_f1_std']:.4f} | "
            f"accuracy={row['accuracy_mean']:.4f}"
        )

    print(f"\nReport: {OUTPUT_DIR / 'cv_tuning_results.json'}")
    print("\n[PASS] Pilot cross-validation completed.")
    print("No final model was selected from test data.")
    print("No existing model checkpoint was overwritten.")


if __name__ == "__main__":
    main()
