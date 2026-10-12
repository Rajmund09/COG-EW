from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(value: str) -> Path:
    """
    Resolve a path relative to the project root when necessary.
    """

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def get_data_root() -> Path:
    """
    Return the configured dataset root.

    Priority:
    1. DATA_ROOT environment variable
    2. data_config.yaml (parsing ${DATA_ROOT:default} if needed)
    3. D:/Datasets (if present)
    4. project/data
    """

    environment_root = os.getenv("DATA_ROOT")

    if environment_root:
        return resolve_path(environment_root)

    config_path = PROJECT_ROOT / "configs" / "data_config.yaml"

    if config_path.exists():

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            config = yaml.safe_load(file) or {}

        configured_root = config.get("data", {}).get("root", "")

        if configured_root.startswith("${"):
            # Format: ${VAR:default}
            inner = configured_root[2:-1]
            parts = inner.split(":", 1)
            var_name = parts[0]
            default_path = parts[1] if len(parts) > 1 else "data"
            
            env_val = os.getenv(var_name)
            if env_val:
                return resolve_path(env_val)
            
            return resolve_path(default_path)

        if configured_root:
            return resolve_path(configured_root)

    d_datasets = Path("D:/Datasets")
    if d_datasets.exists():
        return d_datasets.resolve()

    return PROJECT_ROOT / "data"


def get_device() -> str:
    """
    Determine training device automatically.
    """

    requested_device = os.getenv(
        "DEVICE",
        "auto",
    ).lower()

    if requested_device in {"cpu", "cuda"}:
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"


def get_random_seed() -> int:
    return int(
        os.getenv(
            "RANDOM_SEED",
            "42",
        )
    )


def get_batch_size() -> int:
    return int(
        os.getenv(
            "BATCH_SIZE",
            "32",
        )
    )


def get_num_workers() -> int:
    return int(
        os.getenv(
            "NUM_WORKERS",
            "2",
        )
    )


def get_project_paths() -> dict[str, Path]:

    return {
        "project": PROJECT_ROOT,
        "configs": PROJECT_ROOT / "configs",
        "data": get_data_root(),
        "processed_data": PROJECT_ROOT / "data" / "processed",
        "models": PROJECT_ROOT / "models",
        "reports": PROJECT_ROOT / "reports",
        "logs": PROJECT_ROOT / "logs",
        "dashboard": PROJECT_ROOT / "dashboard",
    }


DATA_ROOT = get_data_root()
DEVICE = get_device()
BATCH_SIZE = get_batch_size()
NUM_WORKERS = get_num_workers()
RANDOM_SEED = get_random_seed()


if __name__ == "__main__":

    print("COG-EW configuration")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data root    : {DATA_ROOT}")
    print(f"Device       : {DEVICE}")
    print(f"Random seed  : {RANDOM_SEED}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {NUM_WORKERS}")

