"""
======================================================================
COG-EW IDF-DS FLIGHT TELEMETRY MODEL TRAINING & GPU BENCHMARK SUITE
(scripts/train_idf_telemetry_models.py)
======================================================================

Executes deep learning training and comprehensive benchmark comparison of
modern temporal flight trajectory & attitude forecasting models on the
Instituto IDF Flight Telemetry Database:

Split Configuration:
  - 60% Training Split    (72 flights: Lap 001 to Lap 072, ~189,198 timesteps)
  - 15% Validation Split  (18 flights: Lap 073 to Lap 090, ~45,021 timesteps)
  - 15% Testing Split     (18 flights: Lap 091 to Lap 108, ~44,540 timesteps)
  - 10% Reserve Split     (12 flights: Lap 109 to Lap 120)

Candidate Architectures Compared:
  1. FlightTransformer   (Multi-Head Self-Attention + Positional Encoding)
  2. TCNResNet           (Dilated 1D Causal ConvNet + Residuals)
  3. BiLSTMAttention     (Bidirectional LSTM + Bahdanau Temporal Attention)
  4. KinematicMLP        (Kinematic Deep MLP Baseline)

Hardware Enforcement:
  - Requires and locks execution to NVIDIA CUDA GPU (e.g. RTX 4050 Laptop GPU).
  - Uses Mixed Precision (AMP FP16) for Tensor Core acceleration.
  - Automatically identifies the winning best model and exports artifacts:
      * models/telemetry/best_telemetry_model.pt
      * reports/experiments/idf_telemetry_comparison_report.json

Usage:
  python scripts/train_idf_telemetry_models.py --device cuda --epochs 10 --batch-size 128
======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telemetry.idf_dataset import (
    create_idf_5fold_cv_dataloaders,
    create_idf_dataloaders,
    get_idf_train_val_test_splits,
)
from telemetry.models import (
    BiLSTMAttention,
    FlightTransformer,
    KinematicMLP,
    TCNResNet,
    create_model,
    get_all_models,
)

MODELS_DIR = PROJECT_ROOT / "models" / "telemetry"
REPORTS_DIR = PROJECT_ROOT / "reports" / "experiments"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def check_and_enforce_gpu(requested_device: str = "cuda") -> torch.device:
    """Validate that NVIDIA CUDA GPU is available and enforce its usage."""
    print("=" * 76)
    print("COG-EW HARDWARE VERIFICATION & ACCELERATOR ENFORCEMENT")
    print("=" * 76)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU is NOT detected by PyTorch!")
        print("This script is strictly configured to execute on the GPU (NVIDIA RTX 4050).")
        print("Please ensure your NVIDIA GPU drivers and PyTorch CUDA build are active.")
        if requested_device == "force-cpu":
            print("[OVERRIDE] --device force-cpu was specified. Continuing on CPU...")
            return torch.device("cpu")
        sys.exit(1)

    device = torch.device(requested_device if requested_device.startswith("cuda") else "cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    cuda_arch = torch.cuda.get_device_capability(device)

    print(f" [PASS] Dedicated GPU Detected : {gpu_name}")
    print(f" [PASS] Dedicated VRAM Memory  : {gpu_mem_gb:.2f} GB")
    print(f" [PASS] Compute Capability     : SM {cuda_arch[0]}.{cuda_arch[1]}")
    print(f" [PASS] PyTorch CUDA Version   : {torch.version.cuda}")
    print(f" [PASS] Execution Target       : {device} (GPU ACTIVE)")
    print("=" * 76)
    return device


def compute_flight_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    features: list[str],
) -> dict[str, float]:
    """
    Compute domain-specific aerospace trajectory and attitude forecast metrics:
      - ADE (Average Displacement Error in meters): Euclidean error across horizon.
      - FDE (Final Displacement Error in meters): Euclidean error at final step.
      - Roll MAE (°), Pitch MAE (°), Yaw MAE (°)
      - Speed MAE (m/s), G-load MAE (g)
    """
    # Features index mapping
    idx_x = features.index("x") if "x" in features else 0
    idx_y = features.index("y") if "y" in features else 1
    idx_z = features.index("z") if "z" in features else 2
    idx_roll = features.index("roll_deg") if "roll_deg" in features else 6
    idx_pitch = features.index("pitch_deg") if "pitch_deg" in features else 7
    idx_yaw = features.index("yaw_deg") if "yaw_deg" in features else 8
    idx_speed = features.index("speed_ms") if "speed_ms" in features else 9
    idx_g = features.index("g_load") if "g_load" in features else 10

    # Displacement errors
    dx = y_pred[:, :, idx_x] - y_true[:, :, idx_x]
    dy = y_pred[:, :, idx_y] - y_true[:, :, idx_y]
    dz = y_pred[:, :, idx_z] - y_true[:, :, idx_z]
    disp_3d = np.sqrt(dx**2 + dy**2 + dz**2)  # (N, H)

    ade = float(np.mean(disp_3d))
    fde = float(np.mean(disp_3d[:, -1]))

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    roll_mae = float(np.mean(np.abs(y_pred[:, :, idx_roll] - y_true[:, :, idx_roll])))
    pitch_mae = float(np.mean(np.abs(y_pred[:, :, idx_pitch] - y_true[:, :, idx_pitch])))
    yaw_mae = float(np.mean(np.abs(y_pred[:, :, idx_yaw] - y_true[:, :, idx_yaw])))
    speed_mae = float(np.mean(np.abs(y_pred[:, :, idx_speed] - y_true[:, :, idx_speed])))
    g_mae = float(np.mean(np.abs(y_pred[:, :, idx_g] - y_true[:, :, idx_g])))

    return {
        "mse": mse,
        "mae": mae,
        "ade_meters": ade,
        "fde_meters": fde,
        "roll_mae_deg": roll_mae,
        "pitch_mae_deg": pitch_mae,
        "yaw_mae_deg": yaw_mae,
        "speed_mae_ms": speed_mae,
        "g_load_mae": g_mae,
    }


def evaluate_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    features: list[str],
    unnormalize_fn: Any = None,
) -> dict[str, float]:
    """Evaluate model on DataLoader and return metric dictionary."""
    model.eval()
    all_preds = []
    all_targets = []

    start_t = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(x)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.numpy())

    total_time = time.perf_counter() - start_t
    y_pred_norm = np.concatenate(all_preds, axis=0)
    y_true_norm = np.concatenate(all_targets, axis=0)

    # Compute normalized space loss for optimization tracking
    norm_mse = float(np.mean((y_pred_norm - y_true_norm) ** 2))
    norm_mae = float(np.mean(np.abs(y_pred_norm - y_true_norm)))

    # Un-normalize to physical domain for aerospace metrics (meters, degrees, m/s)
    if unnormalize_fn is not None:
        y_pred = unnormalize_fn(y_pred_norm)
        y_true = unnormalize_fn(y_true_norm)
    else:
        y_pred = y_pred_norm
        y_true = y_true_norm

    metrics = compute_flight_metrics(y_pred, y_true, features)
    metrics["normalized_mse"] = norm_mse
    metrics["normalized_mae"] = norm_mae
    metrics["latency_ms_per_sample"] = (total_time / len(y_pred)) * 1000.0
    return metrics


def train_single_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    features: list[str],
    device: torch.device,
    unnormalize_fn: Any = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train a single model architecture on the GPU and evaluate across splits."""
    model_name = getattr(model, "model_name", model.__class__.__name__)
    print(f"\n---> Training [{model_name}] on GPU ({device}) | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()  # Huber loss for trajectory outliers
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    training_history = []

    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                preds = model(x)
                loss = criterion(preds, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))

        # Validation (unnormalized physical metrics)
        val_metrics = evaluate_model(model, val_loader, device, features, unnormalize_fn=unnormalize_fn)
        val_loss = val_metrics["normalized_mse"]

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ade": round(val_metrics["ade_meters"], 2),
            "val_roll_mae": round(val_metrics["roll_mae_deg"], 2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "*"
        else:
            marker = " "

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ADE: {val_metrics['ade_meters']:.2f} m | "
            f"Roll MAE: {val_metrics['roll_mae_deg']:.2f}° {marker}"
        )

    total_training_duration = time.perf_counter() - start_train_time

    # Load best checkpoint and evaluate on Test Split
    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_metrics = evaluate_model(model, test_loader, device, features, unnormalize_fn=unnormalize_fn)
    model_checkpoint_path = MODELS_DIR / f"{model_name}.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "metrics": test_metrics,
            "features": features,
            "mean": mean,
            "std": std,
            "params_count": sum(p.numel() for p in model.parameters()),
            "trained_epochs": epochs,
            "training_split": "60% Training (72 IDF Flights)",
        },
        model_checkpoint_path,
    )

    return {
        "model_name": model_name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_time_s": round(total_training_duration, 1),
        "best_val_loss": round(best_val_loss, 4),
        "test_metrics": {
            "mse": round(test_metrics["mse"], 4),
            "mae": round(test_metrics["mae"], 4),
            "ade_meters": round(test_metrics["ade_meters"], 2),
            "fde_meters": round(test_metrics["fde_meters"], 2),
            "roll_mae_deg": round(test_metrics["roll_mae_deg"], 2),
            "pitch_mae_deg": round(test_metrics["pitch_mae_deg"], 2),
            "yaw_mae_deg": round(test_metrics["yaw_mae_deg"], 2),
            "speed_mae_ms": round(test_metrics["speed_mae_ms"], 2),
            "g_load_mae": round(test_metrics["g_load_mae"], 2),
            "latency_ms": round(test_metrics["latency_ms_per_sample"], 3),
        },
        "history": training_history,
        "checkpoint_path": str(model_checkpoint_path),
    }


def run_benchmark_and_select_best(args: argparse.Namespace) -> None:
    """Run model comparison suite, rank architectures, and crown the best model."""
    device = check_and_enforce_gpu(args.device)

    print("\n[INFO] Initializing PyTorch DataLoaders on 60/15/15 Telemetry Split...")
    train_loader, val_loader, test_loader, meta = create_idf_dataloaders(
        batch_size=args.batch_size,
        input_steps=20,
        pred_steps=10,
        num_workers=0,
    )

    print(f" [PASS] 60% Training Split   : {meta['train_samples']:,} sequence windows (Flights 1 to 72)")
    print(f" [PASS] 15% Validation Split : {meta['val_samples']:,} sequence windows (Flights 73 to 90)")
    print(f" [PASS] 15% Testing Split    : {meta['test_samples']:,} sequence windows (Flights 91 to 108)")
    print(f" [PASS] Telemetry Features   : {meta['features']}")

    candidates = get_all_models(
        in_features=len(meta["features"]),
        out_features=len(meta["features"]),
        input_steps=meta["input_steps"],
        pred_steps=meta["pred_steps"],
    )

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 76)
    print("STARTING MULTI-MODEL DEEP LEARNING BENCHMARK COMPARISON ON GPU")
    print("=" * 76)

    for name, model in candidates.items():
        res = train_single_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            features=meta["features"],
            device=device,
            unnormalize_fn=meta.get("unnormalize_fn"),
            mean=meta.get("mean"),
            std=meta.get("std"),
            epochs=args.epochs,
            lr=args.lr,
        )
        results.append(res)

    # Rank models by Test Average Displacement Error (ADE) and Test MSE
    results.sort(key=lambda r: (r["test_metrics"]["ade_meters"], r["test_metrics"]["mse"]))
    best_result = results[0]
    best_name = best_result["model_name"]

    # Copy / Save best model to canonical location
    best_src = Path(best_result["checkpoint_path"])
    best_target = MODELS_DIR / "best_telemetry_model.pt"
    if best_src.exists():
        import shutil
        shutil.copyfile(best_src, best_target)

    # Print Formatted Leaderboard
    print("\n" + "=" * 90)
    print(" INSTITUTO IDF FLIGHT TELEMETRY BENCHMARK LEADERBOARD (60% TRAIN / 15% VAL / 15% TEST)")
    print("=" * 90)
    header = f"{'Rank':<5} | {'Model Architecture':<18} | {'Params':<9} | {'ADE (m)':<8} | {'FDE (m)':<8} | {'Roll MAE':<9} | {'G-MAE':<6} | {'Status'}"
    print(header)
    print("-" * 90)

    for rank, r in enumerate(results, start=1):
        tm = r["test_metrics"]
        status = "★ WINNER (BEST)" if rank == 1 else "COMPLETED"
        print(
            f"{rank:<5} | "
            f"{r['model_name']:<18} | "
            f"{r['parameters']:<9,} | "
            f"{tm['ade_meters']:<8.2f} | "
            f"{tm['fde_meters']:<8.2f} | "
            f"{tm['roll_mae_deg']:<8.2f}° | "
            f"{tm['g_load_mae']:<6.2f} | "
            f"{status}"
        )
    print("=" * 90)
    print(f"\n[SELECTION] BEST MODEL SELECTED: >>> {best_name} <<<")
    print(f" [PASS] Best model weights exported to: {best_target}")

    # ------------------------------------------------------------------
    # 5-FOLD CROSS VALIDATION ON 10% HOLDOUT SPLIT (Flights 109 to 120)
    # ------------------------------------------------------------------
    cv_summary = None
    if not args.skip_cv:
        cv_summary = run_5fold_cross_validation(
            best_model_name=best_name,
            meta=meta,
            device=device,
            cv_epochs=args.cv_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    # Build and write final report
    report = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "Instituto IDF Flight Telemetry (120 Laps)",
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
        },
        "splits_configuration": {
            "training_percent": 60.0,
            "validation_percent": 15.0,
            "testing_percent": 15.0,
            "cross_validation_percent": 10.0,
            "cross_validation_protocol": "5-Fold Stratified Flight Cross-Validation (Lap 109 to 120)",
            "train_samples": meta["train_samples"],
            "val_samples": meta["val_samples"],
            "test_samples": meta["test_samples"],
            "cv_flights_count": len(meta.get("cv_flights", [])),
        },
        "best_model_name": best_name,
        "best_model_metrics": best_result["test_metrics"],
        "comparison_results": results,
        "five_fold_cross_validation": cv_summary,
    }

    report_path = REPORTS_DIR / "idf_telemetry_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n [PASS] Comprehensive benchmark & 5-fold CV report exported to: {report_path}")
    print("Tactical Simulation is configured to use the trained telemetry model & trajectories!")


def run_5fold_cross_validation(
    best_model_name: str,
    meta: dict[str, Any],
    device: torch.device,
    cv_epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """
    Execute 5-Fold Cross Validation on the remaining 10% holdout split (Flights 109 to 120).
    Partitions the 10% split into 5 balanced folds preserving whole flight laps,
    and quantifies mean ± std for trajectory ADE, FDE, and attitude angles.
    """
    print("\n" + "=" * 90)
    print(" INITIATING 5-FOLD CROSS VALIDATION ON 10% HOLDOUT SPLIT (Flights 109 to 120)")
    print("=" * 90)
    print(f"Target Architecture : {best_model_name}")
    print(f"Cross-Val Protocol  : 5 Folds (Whole Flight Laps Preserved, 0 Spatial-Temporal Bleed)")
    print(f"Folds Iterations    : {cv_epochs} epochs per fold on GPU ({device})")
    print("-" * 90)

    cv_loaders = create_idf_5fold_cv_dataloaders(
        cv_flights=meta.get("cv_flights"),
        n_splits=5,
        batch_size=batch_size,
        input_steps=meta["input_steps"],
        pred_steps=meta["pred_steps"],
        mean=meta["mean"],
        std=meta["std"],
    )

    fold_results = []

    for fold_data in cv_loaders:
        f_idx = fold_data["fold"]
        val_laps_str = ", ".join(fold_data["val_laps"])
        print(f"\n---> [Fold {f_idx}/5] Evaluating on Holdout Flights: [{val_laps_str}] | "
              f"Train: {fold_data['train_samples']:,} samples | Val: {fold_data['val_samples']:,} samples")

        fold_model = create_model(
            best_model_name,
            in_features=len(meta["features"]),
            out_features=len(meta["features"]),
            input_steps=meta["input_steps"],
            pred_steps=meta["pred_steps"],
        )

        fold_train_res = train_single_model(
            model=fold_model,
            train_loader=fold_data["train_loader"],
            val_loader=fold_data["val_loader"],
            test_loader=fold_data["val_loader"],
            features=meta["features"],
            device=device,
            unnormalize_fn=meta.get("unnormalize_fn"),
            mean=meta.get("mean"),
            std=meta.get("std"),
            epochs=cv_epochs,
            lr=lr,
        )

        tm = fold_train_res["test_metrics"]
        fold_results.append({
            "fold": f_idx,
            "val_laps": fold_data["val_laps"],
            "train_samples": fold_data["train_samples"],
            "val_samples": fold_data["val_samples"],
            "metrics": tm,
        })

    # Compute Aggregate Mean ± Std across the 5 Folds
    ades = [f["metrics"]["ade_meters"] for f in fold_results]
    fdes = [f["metrics"]["fde_meters"] for f in fold_results]
    rolls = [f["metrics"]["roll_mae_deg"] for f in fold_results]
    pitches = [f["metrics"]["pitch_mae_deg"] for f in fold_results]
    yaws = [f["metrics"]["yaw_mae_deg"] for f in fold_results]
    g_loads = [f["metrics"]["g_load_mae"] for f in fold_results]

    mean_ade = float(np.mean(ades))
    std_ade = float(np.std(ades))
    mean_fde = float(np.mean(fdes))
    std_fde = float(np.std(fdes))
    mean_roll = float(np.mean(rolls))
    std_roll = float(np.std(rolls))
    mean_pitch = float(np.mean(pitches))
    std_pitch = float(np.std(pitches))
    mean_yaw = float(np.mean(yaws))
    std_yaw = float(np.std(yaws))
    mean_g = float(np.mean(g_loads))
    std_g = float(np.std(g_loads))

    # Print 5-Fold Summary Table
    print("\n" + "=" * 90)
    print(" 5-FOLD CROSS VALIDATION SUMMARY TABLE (10% HOLDOUT SPLIT)")
    print("=" * 90)
    print(f"{'Fold':<6} | {'Validation Flights':<26} | {'Val Pts':<8} | {'ADE (m)':<9} | {'FDE (m)':<9} | {'Roll MAE':<10} | {'G-MAE':<6}")
    print("-" * 90)
    for f in fold_results:
        m = f["metrics"]
        laps_preview = ", ".join(f["val_laps"])
        print(f"{f['fold']:<6} | {laps_preview:<26} | {f['val_samples']:<8} | {m['ade_meters']:<9.2f} | {m['fde_meters']:<9.2f} | {m['roll_mae_deg']:<9.2f}° | {m['g_load_mae']:<6.2f}")
    print("-" * 90)
    print(f"CROSS-VALIDATION 5-FOLD MEAN ± STD:")
    print(f"  * Average Displacement Error (ADE) : {mean_ade:.2f} ± {std_ade:.2f} m")
    print(f"  * Final Displacement Error (FDE)   : {mean_fde:.2f} ± {std_fde:.2f} m")
    print(f"  * Attitude Roll MAE (Bank Angle)   : {mean_roll:.2f} ± {std_roll:.2f}°")
    print(f"  * Dynamic G-Load MAE               : {mean_g:.2f} ± {std_g:.2f} g")
    print("=" * 90)

    return {
        "architecture": best_model_name,
        "protocol": "5-Fold Stratified Flight Cross-Validation",
        "dataset_split": "10% Holdout Split (12 flights: Lap 109 to Lap 120)",
        "epochs_per_fold": cv_epochs,
        "aggregate_metrics": {
            "mean_ade_meters": round(mean_ade, 2),
            "std_ade_meters": round(std_ade, 2),
            "mean_fde_meters": round(mean_fde, 2),
            "std_fde_meters": round(std_fde, 2),
            "mean_roll_mae_deg": round(mean_roll, 2),
            "std_roll_mae_deg": round(std_roll, 2),
            "mean_pitch_mae_deg": round(mean_pitch, 2),
            "std_pitch_mae_deg": round(std_pitch, 2),
            "mean_yaw_mae_deg": round(mean_yaw, 2),
            "std_yaw_mae_deg": round(std_yaw, 2),
            "mean_g_load_mae": round(mean_g, 2),
            "std_g_load_mae": round(std_g, 2),
        },
        "folds": fold_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Compare Modern Flight Telemetry Models & Run 5-Fold CV on GPU.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per architecture (default: 10)")
    parser.add_argument("--cv-epochs", type=int, default=3, help="Number of epochs per fold for 5-Fold Cross Validation (default: 3)")
    parser.add_argument("--skip-cv", action="store_true", help="Skip the 5-fold cross-validation suite on 10%% holdout split")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU training (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default="cuda", help="Accelerator target (enforces 'cuda')")
    args = parser.parse_args()

    run_benchmark_and_select_best(args)
