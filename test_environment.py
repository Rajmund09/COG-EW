from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "numpy": "NumPy",
    "pandas": "Pandas",
    "scipy": "SciPy",
    "sklearn": "Scikit-learn",
    "matplotlib": "Matplotlib",
    "plotly": "Plotly",
    "torch": "PyTorch",
    "h5py": "HDF5/h5py",
    "duckdb": "DuckDB",
    "gymnasium": "Gymnasium",
    "stable_baselines3": "Stable-Baselines3",
    "networkx": "NetworkX",
    "streamlit": "Streamlit",
    "yaml": "PyYAML",
    "psutil": "psutil",
    "pytest": "pytest",
}


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_package(module_name: str, display_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")

        print(f"[OK]   {display_name:<24} {version}")
        return True

    except Exception as exc:
        print(f"[FAIL] {display_name:<24} {exc}")
        return False


def check_python() -> bool:
    print_header("PYTHON ENVIRONMENT")

    version_info = sys.version_info

    print(
        f"Python version: "
        f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    )

    print(f"Python executable: {sys.executable}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")

    if version_info < (3, 10):
        print("[WARN] Python 3.10+ is recommended.")

    if version_info >= (3, 10):
        print("[OK]   Python version is supported.")
        return True

    return False


def check_pytorch() -> bool:
    print_header("PYTORCH / GPU")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA compiled into PyTorch: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")

            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                props = torch.cuda.get_device_properties(index)

                vram_gb = props.total_memory / (1024**3)

                print(f"GPU {index}: {name}")
                print(f"VRAM: {vram_gb:.2f} GB")

        else:
            print("[INFO] CUDA GPU not detected.")
            print("[INFO] CPU training will be used initially.")

        return True

    except Exception as exc:
        print(f"[FAIL] PyTorch GPU check failed: {exc}")
        return False


def check_system_resources() -> bool:
    print_header("SYSTEM RESOURCES")

    try:
        import psutil

        cpu_count = os.cpu_count() or 1
        ram_gb = psutil.virtual_memory().total / (1024**3)

        print(f"CPU logical cores: {cpu_count}")
        print(f"System RAM: {ram_gb:.2f} GB")

        return True

    except Exception as exc:
        print(f"[FAIL] Resource detection failed: {exc}")
        return False


def check_project_structure() -> bool:
    print_header("PROJECT STRUCTURE")

    required_directories = [
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "data/loaders",
        "scripts",
        "preprocessing",
        "perception",
        "telemetry",
        "simulation",
        "rl",
        "safety",
        "knowledge_graph",
        "memory",
        "explain",
        "evaluation",
        "dashboard",
        "dashboard/pages",
        "models",
        "reports",
        "reports/figures",
        "logs",
        "tests",
        "notebooks",
    ]

    project_root = Path(__file__).resolve().parent

    missing = []

    for directory in required_directories:
        path = project_root / directory

        if path.exists():
            print(f"[OK]   {directory}")
        else:
            print(f"[MISS] {directory}")
            missing.append(directory)

    if missing:
        print()
        print("[WARN] Some project directories are missing.")
        return False

    return True


def main() -> None:
    print_header("COG-EW ENVIRONMENT DIAGNOSTIC")

    print(f"Project root: {Path(__file__).resolve().parent}")

    python_ok = check_python()

    print_header("REQUIRED PYTHON PACKAGES")

    package_results = []

    for module_name, display_name in REQUIRED_PACKAGES.items():
        package_results.append(check_package(module_name, display_name))

    pytorch_ok = check_pytorch()
    resources_ok = check_system_resources()
    structure_ok = check_project_structure()

    print_header("FINAL STATUS")

    package_ok = all(package_results)

    print(f"Python:          {'PASS' if python_ok else 'FAIL'}")
    print(f"Packages:        {'PASS' if package_ok else 'FAIL'}")
    print(f"PyTorch/GPU:     {'PASS' if pytorch_ok else 'FAIL'}")
    print(f"System resources:{'PASS' if resources_ok else 'FAIL'}")
    print(f"Project folders: {'PASS' if structure_ok else 'FAIL'}")

    if python_ok and package_ok and pytorch_ok and resources_ok:
        print()
        print("COG-EW environment diagnostic completed successfully.")
        print("You can proceed to dataset discovery.")
    else:
        print()
        print("COG-EW environment has unresolved issues.")
        print("Fix the FAILED checks before continuing.")


if __name__ == "__main__":
    main()
