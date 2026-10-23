"""
======================================================================
COG-EW IDF-DS FLIGHT TELEMETRY DATASET & SPLIT MANAGER
(telemetry/idf_dataset.py)
======================================================================

Dataset Source:
  Instituto IDF (Universitat Politècnica de València - UPV)
  UAV Flight Telemetry Database: 120 flights conducted over 3,981 m circuit.
  Path: D:/Datasets/IDF-DS - Flight telemetry

Key Capabilities:
  1. Discovery of IDF dataset in D:/Datasets with unicode-safe path resolution.
  2. Ingestion of 120 Holybro Pixhawk flight laps (position, velocity, quaternions,
     angular rates, accelerations, and aerodynamics).
  3. Partitioning into exact user-specified splits:
     - 60% Training Split    (Flights 1 to 72)
     - 15% Validation Split  (Flights 73 to 90)
     - 15% Testing Split     (Flights 91 to 108)
     - 10% Reserve Split     (Flights 109 to 120)
  4. Preprocessed trajectory caching in data/processed/idf_telemetry/
  5. PyTorch Dataset & DataLoader generation for temporal trajectory forecasting.
======================================================================
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "idf_telemetry"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_idf_dataset_path() -> Path:
    """Locate the IDF-DS Flight Telemetry folder in D:/Datasets with unicode-safe matching."""
    datasets_root = Path("D:/Datasets")
    if not datasets_root.is_dir():
        # Fallback to local data folder if D:/ not mounted
        local_candidate = PROJECT_ROOT / "data" / "raw" / "idf_telemetry"
        if local_candidate.is_dir():
            return local_candidate
        raise FileNotFoundError("Could not find D:/Datasets folder on host system.")

    for entry in os.listdir(str(datasets_root)):
        if "IDF" in entry:
            candidate = datasets_root / entry
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError("Could not find IDF-DS folder inside D:/Datasets.")


def quat_to_euler_deg(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert orientation quaternions (q0, q1, q2, q3) to Euler angles (Roll, Pitch, Yaw) in degrees."""
    # Roll (phi)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # Pitch (theta)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Yaw (psi)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return roll, pitch, yaw


def process_flight_lap(lap_dir: Path, lap_id: int) -> dict[str, Any] | None:
    """Extract and synchronize position, velocity, attitude and aerodynamics for a single lap."""
    files = os.listdir(str(lap_dir))
    pos_file = next((f for f in files if "vehicle_local_position_0" in f), None)
    att_file = next((f for f in files if "vehicle_attitude_0" in f and "setpoint" not in f), None)

    if not pos_file or not att_file:
        return None

    try:
        df_pos = pd.read_csv(lap_dir / pos_file)[["timestamp", "x", "y", "z", "vx", "vy", "vz", "heading"]].dropna()
        df_att = pd.read_csv(lap_dir / att_file)[["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"]].dropna()

        # Merge on nearest timestamp (position is 10Hz, attitude is 20Hz)
        df_merged = pd.merge_asof(
            df_pos.sort_values("timestamp"),
            df_att.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        ).dropna()

        # Compute Euler angles
        w = df_merged["q[0]"].to_numpy()
        x = df_merged["q[1]"].to_numpy()
        y = df_merged["q[2]"].to_numpy()
        z = df_merged["q[3]"].to_numpy()
        roll, pitch, yaw = quat_to_euler_deg(w, x, y, z)

        vx = df_merged["vx"].to_numpy()
        vy = df_merged["vy"].to_numpy()
        vz = df_merged["vz"].to_numpy()
        speed_ms = np.sqrt(vx**2 + vy**2 + vz**2)

        # Acceleration and G-load estimation
        dt = np.gradient(df_merged["timestamp"].to_numpy() / 1e6)
        dt[dt <= 0] = 0.1
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        az = np.gradient(vz, dt)
        g_load = np.sqrt(ax**2 + ay**2 + (az + 9.81)**2) / 9.81

        timestamps = (df_merged["timestamp"].to_numpy() - df_merged["timestamp"].iloc[0]) / 1e6

        return {
            "lap_id": lap_id,
            "lap_name": f"Lap_{lap_id:03d}",
            "num_points": len(df_merged),
            "duration_s": float(timestamps[-1]),
            "timestamps": timestamps.astype(np.float32),
            "x": df_merged["x"].to_numpy().astype(np.float32),
            "y": df_merged["y"].to_numpy().astype(np.float32),
            "z": df_merged["z"].to_numpy().astype(np.float32),
            "vx": vx.astype(np.float32),
            "vy": vy.astype(np.float32),
            "vz": vz.astype(np.float32),
            "speed_ms": speed_ms.astype(np.float32),
            "roll_deg": roll.astype(np.float32),
            "pitch_deg": pitch.astype(np.float32),
            "yaw_deg": yaw.astype(np.float32),
            "g_load": g_load.astype(np.float32),
            "max_roll": float(np.max(np.abs(roll))),
            "max_speed": float(np.max(speed_ms)),
            "peak_g": float(np.max(g_load)),
        }
    except Exception as e:
        print(f"[WARNING] Error parsing lap {lap_id}: {e}")
        return None


def get_all_flight_trajectories(cache_file: Path | None = None) -> list[dict[str, Any]]:
    """Load or build the consolidated 120-flight trajectory database."""
    if cache_file is None:
        cache_file = PROCESSED_DATA_DIR / "idf_120_flights.npz"

    # If cached file exists, load it
    if cache_file.is_file():
        try:
            data = np.load(cache_file, allow_pickle=True)
            flights = data["flights"].tolist()
            if len(flights) >= 100:
                return flights
        except Exception:
            pass

    # Otherwise process from raw IDF folder
    idf_path = find_idf_dataset_path()
    ungrouped = idf_path / "Holybro Pixhawk" / "processed" / "UnGrouped flights"
    if not ungrouped.is_dir():
        raise FileNotFoundError(f"Missing UnGrouped flights directory in {idf_path}")

    lap_dirs = sorted([d for d in os.listdir(str(ungrouped)) if os.path.isdir(str(ungrouped / d))])
    flights = []

    print(f"[INFO] Ingesting {len(lap_dirs)} real flight telemetry laps from IDF dataset...")
    for idx, d_name in enumerate(lap_dirs, start=1):
        lap_dir = ungrouped / d_name
        flight = process_flight_lap(lap_dir, lap_id=idx)
        if flight is not None:
            flights.append(flight)

    print(f"[INFO] Successfully processed {len(flights)} flight telemetry records.")

    # Save to compressed cache
    np.savez_compressed(cache_file, flights=np.array(flights, dtype=object))
    return flights


def get_idf_train_val_test_splits(
    train_pct: float = 0.60,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split the 120 flight telemetry dataset into:
      - 60% Training Split    (Flights 1 to 72)
      - 15% Validation Split  (Flights 73 to 90)
      - 15% Testing Split     (Flights 91 to 108)
      - 10% Reserve Split     (Flights 109 to 120)
    """
    flights = get_all_flight_trajectories()
    n = len(flights)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)

    train_flights = flights[:n_train]
    val_flights = flights[n_train : n_train + n_val]
    test_flights = flights[n_train + n_val : n_train + n_val + n_test]
    reserve_flights = flights[n_train + n_val + n_test :]

    # Save split catalog manifest
    manifest = {
        "dataset_name": "Instituto IDF UAV Flight Telemetry Database",
        "dataset_path": str(find_idf_dataset_path()),
        "total_flights": n,
        "split_configuration": {
            "training_percent": round(train_pct * 100, 1),
            "validation_percent": round(val_pct * 100, 1),
            "testing_percent": round(test_pct * 100, 1),
            "reserve_percent": round((1.0 - train_pct - val_pct - test_pct) * 100, 1),
        },
        "counts": {
            "train_flights_count": len(train_flights),
            "val_flights_count": len(val_flights),
            "test_flights_count": len(test_flights),
            "reserve_flights_count": len(reserve_flights),
            "train_samples_count": sum(f["num_points"] for f in train_flights),
            "val_samples_count": sum(f["num_points"] for f in val_flights),
            "test_samples_count": sum(f["num_points"] for f in test_flights),
        },
        "train_laps": [f["lap_name"] for f in train_flights],
        "val_laps": [f["lap_name"] for f in val_flights],
        "test_laps": [f["lap_name"] for f in test_flights],
    }

    manifest_file = PROCESSED_DATA_DIR / "idf_splits_60_15_15_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "train": train_flights,
        "val": val_flights,
        "test": test_flights,
        "reserve": reserve_flights,
    }


class IDFTelemetrySequenceDataset(Dataset):
    """
    Sliding-window trajectory sequence dataset for training modern temporal models.
    Inputs: Past K steps (e.g. 20 timesteps = 2.0s history at 10Hz)
    Targets: Future H steps (e.g. 10 timesteps = 1.0s prediction)
    Features: [x, y, z, vx, vy, vz, roll, pitch, yaw, speed_ms, g_load]
    """

    def __init__(
        self,
        flights: list[dict[str, Any]],
        input_steps: int = 20,
        pred_steps: int = 10,
        stride: int = 2,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.stride = stride

        # Feature normalization statistics (mean and std)
        self.feature_names = ["x", "y", "z", "vx", "vy", "vz", "roll_deg", "pitch_deg", "yaw_deg", "speed_ms", "g_load"]
        self.samples_x: list[np.ndarray] = []
        self.samples_y: list[np.ndarray] = []

        self._build_sequences(flights)

        # Compute or apply standardization statistics
        if mean is None or std is None:
            # Compute across all raw input sequence timesteps
            flat_x = self.samples_x.reshape(-1, len(self.feature_names))
            self.mean = np.mean(flat_x, axis=0).astype(np.float32)
            self.std = (np.std(flat_x, axis=0) + 1e-4).astype(np.float32)
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

    def _build_sequences(self, flights: list[dict[str, Any]]):
        seq_len = self.input_steps + self.pred_steps
        for f in flights:
            pts = f["num_points"]
            if pts < seq_len:
                continue

            # Stack features: shape (pts, num_features)
            mat = np.column_stack([f[k] for k in self.feature_names]).astype(np.float32)
            # Guarantee zero NaNs or Infs
            mat = np.nan_to_num(mat, nan=0.0, posinf=5.0, neginf=-5.0)

            for start in range(0, pts - seq_len + 1, self.stride):
                x_seq = mat[start : start + self.input_steps]
                y_seq = mat[start + self.input_steps : start + seq_len]
                self.samples_x.append(x_seq)
                self.samples_y.append(y_seq)

        self.samples_x = np.array(self.samples_x, dtype=np.float32)
        self.samples_y = np.array(self.samples_y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples_x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return standardized tensors for smooth gradient descent
        x_norm = (self.samples_x[idx] - self.mean) / self.std
        y_norm = (self.samples_y[idx] - self.mean) / self.std
        return torch.from_numpy(x_norm), torch.from_numpy(y_norm)

    def unnormalize(self, norm_arr: np.ndarray) -> np.ndarray:
        """Denormalize network outputs back to physical units (meters, degrees, m/s)."""
        return norm_arr * self.std + self.mean


def create_idf_dataloaders(
    batch_size: int = 64,
    input_steps: int = 20,
    pred_steps: int = 10,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Create GPU-ready PyTorch DataLoaders for 60% Train, 15% Val, and 15% Test splits with standardization."""
    splits = get_idf_train_val_test_splits(train_pct=0.60, val_pct=0.15, test_pct=0.15)

    train_ds = IDFTelemetrySequenceDataset(splits["train"], input_steps=input_steps, pred_steps=pred_steps, stride=2)
    val_ds = IDFTelemetrySequenceDataset(splits["val"], input_steps=input_steps, pred_steps=pred_steps, stride=4, mean=train_ds.mean, std=train_ds.std)
    test_ds = IDFTelemetrySequenceDataset(splits["test"], input_steps=input_steps, pred_steps=pred_steps, stride=4, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    metadata = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "input_steps": input_steps,
        "pred_steps": pred_steps,
        "features": train_ds.feature_names,
        "mean": train_ds.mean,
        "std": train_ds.std,
        "unnormalize_fn": train_ds.unnormalize,
        "cv_flights": splits["reserve"],
        "splits": splits,
    }

    return train_loader, val_loader, test_loader, metadata


def get_idf_5fold_cv_splits(
    cv_flights: list[dict[str, Any]] | None = None,
    n_splits: int = 5,
) -> list[dict[str, Any]]:
    """
    Partition the 10% holdout split (12 flights: Lap 109 to 120, ~31,000 samples)
    into 5 balanced cross-validation folds preserving whole flight trajectories.
    """
    if cv_flights is None:
        splits = get_idf_train_val_test_splits(train_pct=0.60, val_pct=0.15, test_pct=0.15)
        cv_flights = splits["reserve"]

    n_flights = len(cv_flights)
    indices = np.arange(n_flights)
    fold_splits = np.array_split(indices, n_splits)

    cv_folds = []
    for fold_idx, val_idx in enumerate(fold_splits, start=1):
        val_idx_set = set(val_idx.tolist())
        train_idx = [i for i in range(n_flights) if i not in val_idx_set]

        train_sub = [cv_flights[i] for i in train_idx]
        val_sub = [cv_flights[i] for i in val_idx]

        cv_folds.append({
            "fold": fold_idx,
            "train_flights": train_sub,
            "val_flights": val_sub,
            "train_laps": [f["lap_name"] for f in train_sub],
            "val_laps": [f["lap_name"] for f in val_sub],
            "train_points": sum(f["num_points"] for f in train_sub),
            "val_points": sum(f["num_points"] for f in val_sub),
        })

    return cv_folds


def create_idf_5fold_cv_dataloaders(
    cv_flights: list[dict[str, Any]] | None = None,
    n_splits: int = 5,
    batch_size: int = 64,
    input_steps: int = 20,
    pred_steps: int = 10,
    stride_train: int = 2,
    stride_val: int = 4,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    """
    Builds GPU DataLoaders for each of the 5 cross-validation folds on the 10% holdout split.
    """
    folds = get_idf_5fold_cv_splits(cv_flights=cv_flights, n_splits=n_splits)
    cv_loaders = []

    for fold_info in folds:
        train_ds = IDFTelemetrySequenceDataset(
            fold_info["train_flights"],
            input_steps=input_steps,
            pred_steps=pred_steps,
            stride=stride_train,
            mean=mean,
            std=std,
        )
        fold_mean = train_ds.mean if mean is None else mean
        fold_std = train_ds.std if std is None else std

        val_ds = IDFTelemetrySequenceDataset(
            fold_info["val_flights"],
            input_steps=input_steps,
            pred_steps=pred_steps,
            stride=stride_val,
            mean=fold_mean,
            std=fold_std,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        cv_loaders.append({
            "fold": fold_info["fold"],
            "train_loader": train_loader,
            "val_loader": val_loader,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "train_laps": fold_info["train_laps"],
            "val_laps": fold_info["val_laps"],
            "features": train_ds.feature_names,
            "mean": fold_mean,
            "std": fold_std,
            "unnormalize_fn": val_ds.unnormalize,
        })

    return cv_loaders


if __name__ == "__main__":
    print("=" * 72)
    print("IDF FLIGHT TELEMETRY DATASET & 60/15/15 SPLIT INSPECTION")
    print("=" * 72)
    path = find_idf_dataset_path()
    clean_p = str(path).encode("ascii", "ignore").decode("ascii")
    print("Discovered Path:", clean_p)

    splits = get_idf_train_val_test_splits(train_pct=0.60, val_pct=0.15, test_pct=0.15)
    print(f"60% Training Split:   {len(splits['train'])} flights ({sum(f['num_points'] for f in splits['train']):,} points)")
    print(f"15% Validation Split: {len(splits['val'])} flights ({sum(f['num_points'] for f in splits['val']):,} points)")
    print(f"15% Testing Split:    {len(splits['test'])} flights ({sum(f['num_points'] for f in splits['test']):,} points)")
    print(f"10% Reserve Split:    {len(splits['reserve'])} flights")
    print("Sample Flight 01:")
    f1 = splits['train'][0]
    print(f"  Duration: {f1['duration_s']:.1f} s | Points: {f1['num_points']} | Max Roll: {f1['max_roll']:.1f}° | Max Speed: {f1['max_speed']:.1f} m/s")
    print("=" * 72)
