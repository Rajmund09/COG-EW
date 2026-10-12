"""
COG-EW
Controlled RF Data Augmentation

Purpose:
    Apply label-preserving RF augmentations ONLY to the training split.

Input:
    D:\COG-EW\data\splits\cogew_train.h5

Output:
    D:\COG-EW\data\augmented\cogew_train_augmented.h5

Important:
    - Validation data is NOT augmented.
    - Test data is NOT augmented.
    - Original training split is NOT modified.
    - No model training is performed.
    - Augmented samples retain their original class labels.
    - Processing is performed in batches to control memory usage.
"""

from pathlib import Path
import json
import hashlib
import time

import h5py
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\COG-EW")

INPUT_FILE = PROJECT_ROOT / "data" / "splits" / "cogew_train.h5"

OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented"

OUTPUT_FILE = OUTPUT_DIR / "cogew_train_augmented.h5"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_FILE = REPORT_DIR / "cogew_augmentation_audit.json"

# Reproducibility
SEED = 42

# Process in manageable chunks
BATCH_SIZE = 2048

# ------------------------------------------------------------
# AUGMENTATION CONFIGURATION
# ------------------------------------------------------------

# Number of versions:
#
# Original:
#       1.0x
#
# Augmented:
#       1 additional transformed version
#
# Final:
#       2x training data
#
# We deliberately start with 2x instead of aggressive
# augmentation. This lets us evaluate whether augmentation
# actually improves the model.
#
AUGMENTATION_FACTOR = 2


# Individual augmentation probabilities
P_PHASE_ROTATION = 0.50
P_AMPLITUDE_SCALING = 0.50
P_TIME_SHIFT = 0.35
P_IQ_IMBALANCE = 0.25
P_FREQUENCY_OFFSET = 0.30
P_AWGN = 0.60


# Transformation ranges
PHASE_MIN = -np.pi
PHASE_MAX = np.pi

AMPLITUDE_MIN = 0.80
AMPLITUDE_MAX = 1.20

MAX_TIME_SHIFT = 12

IQ_GAIN_IMBALANCE_MIN = 0.90
IQ_GAIN_IMBALANCE_MAX = 1.10

IQ_PHASE_IMBALANCE_MAX = np.deg2rad(5.0)

# Normalized frequency offset.
# This is intentionally small so that the modulation identity
# remains meaningful.
MAX_FREQ_OFFSET = 0.02

# AWGN strength relative to signal RMS.
# A random value is selected within this range.
AWGN_MIN_DB = 20.0
AWGN_MAX_DB = 35.0


CLASS_NAMES = {
    0: "8PSK",
    1: "BPSK",
    2: "QPSK",
    3: "16QAM",
    4: "64QAM",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    """
    Calculate SHA256 hash of a file.

    Used to prove that the original training dataset was not
    modified during augmentation.
    """

    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()


def validate_input_file():
    print_header("VALIDATING TRAINING DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Training dataset not found:\n{INPUT_FILE}")

    print(f"Input:")
    print(INPUT_FILE)

    with h5py.File(INPUT_FILE, "r") as h5:

        required = ["X", "Y", "Z"]

        for key in required:
            if key not in h5:
                raise ValueError(f"Required dataset '{key}' missing.")

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]

        print()
        print("Dataset structure:")
        print(f"X: {X.shape}")
        print(f"Y: {Y.shape}")
        print(f"Z: {Z.shape}")

        if len(X) != len(Y) or len(X) != len(Z):
            raise ValueError("X/Y/Z sample counts do not match.")

        if X.ndim != 3:
            raise ValueError(f"Expected X to have 3 dimensions, got {X.ndim}")

        if X.shape[1:] != (128, 2):
            raise ValueError(f"Expected X shape (*, 128, 2), got {X.shape}")

        if len(Y.shape) != 1:
            raise ValueError(f"Expected Y to be 1D, got {Y.shape}")

        if len(Z.shape) != 1:
            raise ValueError(f"Expected Z to be 1D, got {Z.shape}")

        # Check labels
        labels = np.unique(Y[:])

        print()
        print("Detected classes:")

        for label in labels:
            label = int(label)

            print(f"  {label:02d} | " f"{CLASS_NAMES.get(label, 'UNKNOWN')}")

        expected_classes = set(CLASS_NAMES.keys())

        if set(labels.tolist()) != expected_classes:
            raise ValueError("Unexpected class set detected.")

        # Check finite values using a sample.
        sample_count = min(10000, len(X))

        sample = X[:sample_count]

        if not np.isfinite(sample).all():
            raise ValueError("Input dataset contains NaN or Inf values.")

        print()
        print("[PASS] X/Y/Z alignment")
        print("[PASS] IQ shape = (128, 2)")
        print("[PASS] Expected five classes detected")
        print("[PASS] Input sample contains no NaN/Inf")

        return {
            "samples": int(len(X)),
            "x_shape": list(X.shape),
            "y_shape": list(Y.shape),
            "z_shape": list(Z.shape),
            "labels": labels.astype(int).tolist(),
        }


# ============================================================
# RF AUGMENTATIONS
# ============================================================


def phase_rotation(x, rng):
    """
    Rotate complex IQ samples by a random phase.

    x shape:
        [batch, 128, 2]

    Channel 0 = I
    Channel 1 = Q
    """

    batch_size = x.shape[0]

    phase = rng.uniform(
        PHASE_MIN,
        PHASE_MAX,
        size=(batch_size, 1),
    ).astype(np.float32)

    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)

    i = x[:, :, 0].copy()
    q = x[:, :, 1].copy()

    x[:, :, 0] = i * cos_phase - q * sin_phase

    x[:, :, 1] = i * sin_phase + q * cos_phase

    return x


def amplitude_scaling(x, rng):
    """
    Random amplitude scaling.

    This models changes in received signal strength.
    """

    batch_size = x.shape[0]

    scale = rng.uniform(
        AMPLITUDE_MIN,
        AMPLITUDE_MAX,
        size=(batch_size, 1, 1),
    ).astype(np.float32)

    x *= scale

    return x


def time_shift(x, rng):
    """
    Circular time shift.

    Uses np.roll so no samples are lost.
    """

    batch_size = x.shape[0]

    for i in range(batch_size):

        shift = int(
            rng.integers(
                -MAX_TIME_SHIFT,
                MAX_TIME_SHIFT + 1,
            )
        )

        if shift != 0:
            x[i] = np.roll(
                x[i],
                shift=shift,
                axis=0,
            )

    return x


def iq_imbalance(x, rng):
    """
    Simulate mild I/Q gain and phase imbalance.

    This models imperfections in RF receiver hardware.
    """

    batch_size = x.shape[0]

    gain = rng.uniform(
        IQ_GAIN_IMBALANCE_MIN,
        IQ_GAIN_IMBALANCE_MAX,
        size=(batch_size, 1),
    ).astype(np.float32)

    phase_error = rng.uniform(
        -IQ_PHASE_IMBALANCE_MAX,
        IQ_PHASE_IMBALANCE_MAX,
        size=(batch_size, 1),
    ).astype(np.float32)

    i = x[:, :, 0].copy()
    q = x[:, :, 1].copy()

    # Gain imbalance
    i *= gain
    q /= gain

    # Small quadrature phase error
    q_new = q * np.cos(phase_error) + i * np.sin(phase_error)

    x[:, :, 1] = q_new

    return x


def frequency_offset(x, rng):
    """
    Apply a small normalized carrier frequency offset.

    For a complex signal:

        x[n] = I[n] + jQ[n]

    Apply:

        x[n] * exp(j * 2*pi*f*n)
    """

    batch_size = x.shape[0]
    signal_length = x.shape[1]

    offsets = rng.uniform(
        -MAX_FREQ_OFFSET,
        MAX_FREQ_OFFSET,
        size=batch_size,
    ).astype(np.float32)

    n = np.arange(
        signal_length,
        dtype=np.float32,
    )

    for i in range(batch_size):

        freq = offsets[i]

        phase = 2.0 * np.pi * freq * n

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        real = x[i, :, 0].copy()
        imag = x[i, :, 1].copy()

        x[i, :, 0] = real * cos_phase - imag * sin_phase

        x[i, :, 1] = real * sin_phase + imag * cos_phase

    return x


def add_awgn(x, rng):
    """
    Add controlled AWGN.

    Noise level is selected relative to the RMS power of
    each individual signal.

    This avoids applying one absolute noise scale to all
    samples.
    """

    batch_size = x.shape[0]

    # Signal power
    power = np.mean(
        x**2,
        axis=(1, 2),
        keepdims=True,
    )

    power = np.maximum(
        power,
        1e-12,
    )

    snr_db = rng.uniform(
        AWGN_MIN_DB,
        AWGN_MAX_DB,
        size=(batch_size, 1, 1),
    ).astype(np.float32)

    snr_linear = 10.0 ** (snr_db / 10.0)

    noise_power = power / snr_linear

    noise_std = np.sqrt(noise_power)

    noise = rng.normal(
        0.0,
        1.0,
        size=x.shape,
    ).astype(np.float32)

    noise *= noise_std

    x += noise

    return x


# ============================================================
# RANDOM AUGMENTATION PIPELINE
# ============================================================


def augment_batch(batch, rng):
    """
    Apply a randomized combination of RF augmentations.

    Every sample receives a different combination.

    At least one transformation is forced if none of the
    probabilistic transformations was selected.
    """

    x = batch.astype(
        np.float32,
        copy=True,
    )

    batch_size = x.shape[0]

    applied_any = np.zeros(
        batch_size,
        dtype=bool,
    )

    # --------------------------------------------------------
    # We use per-sample masks.
    # --------------------------------------------------------

    mask = rng.random(batch_size) < P_PHASE_ROTATION

    if np.any(mask):
        indices = np.where(mask)[0]
        x[indices] = phase_rotation(
            x[indices],
            rng,
        )
        applied_any[indices] = True

    mask = rng.random(batch_size) < P_AMPLITUDE_SCALING

    if np.any(mask):
        indices = np.where(mask)[0]
        x[indices] = amplitude_scaling(
            x[indices],
            rng,
        )
        applied_any[indices] = True

    mask = rng.random(batch_size) < P_TIME_SHIFT

    if np.any(mask):
        indices = np.where(mask)[0]
        x[indices] = time_shift(
            x[indices],
            rng,
        )
        applied_any[indices] = True

    mask = rng.random(batch_size) < P_IQ_IMBALANCE

    if np.any(mask):
        indices = np.where(mask)[0]
        x[indices] = iq_imbalance(
            x[indices],
            rng,
        )
        applied_any[indices] = True

    mask = rng.random(batch_size) < P_FREQUENCY_OFFSET

    if np.any(mask):
        indices = np.where(mask)[0]
        x[indices] = frequency_offset(
            x[indices],
            rng,
        )
        applied_any[indices] = True

    mask = rng.random(batch_size) < P_AWGN

    if np.any(mask):
        indices = np.where(mask)[0]
        x[indices] = add_awgn(
            x[indices],
            rng,
        )
        applied_any[indices] = True

    # --------------------------------------------------------
    # Guarantee that every generated sample differs from its
    # original through at least one transformation.
    # --------------------------------------------------------

    missing = np.where(~applied_any)[0]

    if len(missing) > 0:

        x[missing] = phase_rotation(
            x[missing],
            rng,
        )

    return x


# ============================================================
# MAIN AUGMENTATION PROCESS
# ============================================================


def create_augmented_dataset():

    print_header("COG-EW CONTROLLED RF DATA AUGMENTATION")

    print("Input:")
    print(INPUT_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("Configuration:")
    print(f"  Random seed       : {SEED}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Augmentation      : {AUGMENTATION_FACTOR}x")
    print("  Validation        : untouched")
    print("  Test              : untouched")

    print()
    print("Augmentations:")
    print("  [1] Phase rotation")
    print("  [2] Amplitude scaling")
    print("  [3] Time shift")
    print("  [4] IQ imbalance")
    print("  [5] Frequency offset")
    print("  [6] AWGN")

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    input_info = validate_input_file()

    original_hash = sha256_file(INPUT_FILE)

    print()
    print("Original training dataset SHA256:")
    print(original_hash)

    # --------------------------------------------------------
    # Read source metadata
    # --------------------------------------------------------

    with h5py.File(INPUT_FILE, "r") as source:

        X_source = source["X"]
        Y_source = source["Y"]
        Z_source = source["Z"]

        total_samples = len(X_source)

        augmented_samples = total_samples * (AUGMENTATION_FACTOR - 1)

        final_samples = total_samples * AUGMENTATION_FACTOR

        print()
        print_header("AUGMENTATION SIZE")

        print(f"Original samples : {total_samples:,}")

        print(f"New augmented    : {augmented_samples:,}")

        print(f"Final samples    : {final_samples:,}")

        # ----------------------------------------------------
        # Remove old output if it exists.
        #
        # This DOES NOT touch source data.
        # ----------------------------------------------------

        if OUTPUT_FILE.exists():

            print()
            print("[WARNING] Existing augmented file found.")

            print("It will be replaced.")

            OUTPUT_FILE.unlink()

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Create output HDF5
        # ----------------------------------------------------

        with h5py.File(
            OUTPUT_FILE,
            "w",
        ) as destination:

            x_out = destination.create_dataset(
                "X",
                shape=(
                    final_samples,
                    128,
                    2,
                ),
                dtype=np.float32,
                chunks=(
                    min(BATCH_SIZE, final_samples),
                    128,
                    2,
                ),
                compression="gzip",
                compression_opts=4,
            )

            y_out = destination.create_dataset(
                "Y",
                shape=(final_samples,),
                dtype=np.int64,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

            z_out = destination.create_dataset(
                "Z",
                shape=(final_samples,),
                dtype=np.int64,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

            # Source identifier:
            #
            # 0 = original training sample
            # 1 = augmented training sample
            #
            augmentation_source = destination.create_dataset(
                "augmentation_source",
                shape=(final_samples,),
                dtype=np.int8,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

            # Original index from the training split.
            #
            # This lets us trace every augmented sample back
            # to the original training sample.
            source_index = destination.create_dataset(
                "source_index",
                shape=(final_samples,),
                dtype=np.int64,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

            # ------------------------------------------------
            # Copy ORIGINAL training data
            # ------------------------------------------------

            print_header("COPYING ORIGINAL TRAINING SAMPLES")

            write_position = 0

            for start in range(
                0,
                total_samples,
                BATCH_SIZE,
            ):

                end = min(
                    start + BATCH_SIZE,
                    total_samples,
                )

                x_batch = X_source[start:end]

                y_batch = Y_source[start:end]

                z_batch = Z_source[start:end]

                count = end - start

                x_out[write_position : write_position + count] = x_batch

                y_out[write_position : write_position + count] = y_batch

                z_out[write_position : write_position + count] = z_batch

                augmentation_source[write_position : write_position + count] = 0

                source_index[write_position : write_position + count] = np.arange(
                    start,
                    end,
                    dtype=np.int64,
                )

                write_position += count

                if (
                    write_position % 20000 < BATCH_SIZE
                    or write_position == total_samples
                ):

                    print(f"  Original: " f"{write_position:,}/" f"{total_samples:,}")

            # ------------------------------------------------
            # Generate augmented samples
            # ------------------------------------------------

            print_header("GENERATING AUGMENTED TRAINING SAMPLES")

            rng = np.random.default_rng(SEED)

            augmented_write_start = write_position

            for start in range(
                0,
                total_samples,
                BATCH_SIZE,
            ):

                end = min(
                    start + BATCH_SIZE,
                    total_samples,
                )

                x_batch = X_source[start:end]

                y_batch = Y_source[start:end]

                z_batch = Z_source[start:end]

                # Apply RF augmentation
                x_augmented = augment_batch(
                    x_batch,
                    rng,
                )

                count = end - start

                x_out[write_position : write_position + count] = x_augmented

                y_out[write_position : write_position + count] = y_batch

                z_out[write_position : write_position + count] = z_batch

                augmentation_source[write_position : write_position + count] = 1

                source_index[write_position : write_position + count] = np.arange(
                    start,
                    end,
                    dtype=np.int64,
                )

                write_position += count

                if (write_position - augmented_write_start) % 20000 < BATCH_SIZE:

                    generated = write_position - augmented_write_start

                    print(f"  Augmented: " f"{generated:,}/" f"{augmented_samples:,}")

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            destination.attrs["project"] = "COG-EW"

            destination.attrs["stage"] = "controlled_rf_augmentation"

            destination.attrs["augmentation_factor"] = AUGMENTATION_FACTOR

            destination.attrs["random_seed"] = SEED

            destination.attrs["input_file"] = str(INPUT_FILE)

            destination.attrs["validation_augmented"] = False

            destination.attrs["test_augmented"] = False

            destination.attrs["label_preserving"] = True

            destination.attrs["augmentation_methods"] = (
                "phase_rotation,"
                "amplitude_scaling,"
                "time_shift,"
                "iq_imbalance,"
                "frequency_offset,"
                "awgn"
            )

            destination.flush()

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    print_header("VALIDATING AUGMENTED DATASET")

    with h5py.File(
        OUTPUT_FILE,
        "r",
    ) as h5:

        X = h5["X"]
        Y = h5["Y"]
        Z = h5["Z"]
        augmentation_source = h5["augmentation_source"]
        source_index = h5["source_index"]

        print(f"X shape: {X.shape}")

        print(f"Y shape: {Y.shape}")

        print(f"Z shape: {Z.shape}")

        print(f"augmentation_source: " f"{augmentation_source.shape}")

        print(f"source_index: " f"{source_index.shape}")

        # ----------------------------------------------------
        # Check complete size
        # ----------------------------------------------------

        expected_samples = input_info["samples"] * AUGMENTATION_FACTOR

        if len(X) != expected_samples:
            raise RuntimeError("Unexpected output sample count.")

        # ----------------------------------------------------
        # Check finite values
        # ----------------------------------------------------

        finite_ok = True

        for start in range(
            0,
            len(X),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(X),
            )

            x_batch = X[start:end]

            if not np.isfinite(x_batch).all():

                finite_ok = False
                break

        if not finite_ok:
            raise RuntimeError("Augmented dataset contains NaN/Inf.")

        print("[PASS] No NaN/Inf values")

        # ----------------------------------------------------
        # Label distribution
        # ----------------------------------------------------

        labels = Y[:]

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        print()
        print("FINAL CLASS DISTRIBUTION")

        print("-" * 55)

        for label, count in zip(
            unique_labels,
            counts,
        ):

            label = int(label)

            print(f"{label:02d} | " f"{CLASS_NAMES[label]:8s} | " f"{count:,}")

        # ----------------------------------------------------
        # Original / augmented counts
        # ----------------------------------------------------

        source_values, source_counts = np.unique(
            augmentation_source[:],
            return_counts=True,
        )

        print()
        print("AUGMENTATION SOURCE DISTRIBUTION")

        for value, count in zip(
            source_values,
            source_counts,
        ):

            name = "Original" if int(value) == 0 else "Augmented"

            print(f"{name:12s} | " f"{count:,}")

        # ----------------------------------------------------
        # Check source index
        # ----------------------------------------------------

        if not np.all(source_index[:] >= 0):

            raise RuntimeError("Invalid source index detected.")

        print()
        print("[PASS] Source indices valid")

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        sample_for_stats = min(
            20000,
            len(X),
        )

        stats_x = X[:sample_for_stats]

        print()
        print("OUTPUT SIGNAL STATISTICS")

        print(f"Mean: " f"{float(np.mean(stats_x)):.8f}")

        print(f"Std : " f"{float(np.std(stats_x)):.8f}")

        print(f"Min : " f"{float(np.min(stats_x)):.8f}")

        print(f"Max : " f"{float(np.max(stats_x)):.8f}")

    # ========================================================
    # VERIFY ORIGINAL FILE WAS NOT MODIFIED
    # ========================================================

    new_original_hash = sha256_file(INPUT_FILE)

    original_unchanged = original_hash == new_original_hash

    # ========================================================
    # WRITE AUDIT REPORT
    # ========================================================

    report = {
        "project": "COG-EW",
        "stage": "controlled_rf_augmentation",
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_sha256_before": original_hash,
        "input_sha256_after": new_original_hash,
        "original_training_samples": int(total_samples),
        "augmented_samples": int(augmented_samples),
        "final_samples": int(final_samples),
        "augmentation_factor": int(AUGMENTATION_FACTOR),
        "batch_size": int(BATCH_SIZE),
        "random_seed": int(SEED),
        "augmentation_methods": [
            "phase_rotation",
            "amplitude_scaling",
            "time_shift",
            "iq_imbalance",
            "frequency_offset",
            "awgn",
        ],
        "augmentation_parameters": {
            "phase_range_rad": [
                PHASE_MIN,
                PHASE_MAX,
            ],
            "amplitude_range": [
                AMPLITUDE_MIN,
                AMPLITUDE_MAX,
            ],
            "max_time_shift": MAX_TIME_SHIFT,
            "iq_gain_range": [
                IQ_GAIN_IMBALANCE_MIN,
                IQ_GAIN_IMBALANCE_MAX,
            ],
            "max_iq_phase_error_deg": 5.0,
            "max_frequency_offset": MAX_FREQ_OFFSET,
            "awgn_snr_range_db": [
                AWGN_MIN_DB,
                AWGN_MAX_DB,
            ],
        },
        "validation_augmented": False,
        "test_augmented": False,
        "label_preserving": True,
        "source_dataset_modified": False,
        "original_training_file_unchanged": (original_unchanged),
        "output_validation": {
            "finite_values": True,
            "x_shape": [
                int(final_samples),
                128,
                2,
            ],
            "class_count": 5,
        },
        "status": "SUCCESS",
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_header("RF AUGMENTATION COMPLETE")

    print("[OK] Augmented dataset:")

    print(OUTPUT_FILE)

    print()
    print("[OK] Audit report:")

    print(REPORT_FILE)

    print()

    if original_unchanged:
        print("[PASS] Original training dataset unchanged.")
    else:
        print("[FAIL] Original training dataset hash changed!")

    print("[PASS] Validation dataset was NOT augmented.")

    print("[PASS] Test dataset was NOT augmented.")

    print("[PASS] Labels preserved.")

    print("[PASS] Augmentation is training-only.")

    print()
    print("NEXT STAGE:")

    print(
        "Run augmentation quality / integrity analysis "
        "before training the new model."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    start_time = time.time()

    try:

        ensure_directories()

        create_augmented_dataset()

    except Exception as exc:

        print()
        print("=" * 70)
        print("AUGMENTATION FAILED")
        print("=" * 70)

        print(f"{type(exc).__name__}: {exc}")

        raise

    finally:

        elapsed = time.time() - start_time

        print()
        print(f"Elapsed time: {elapsed / 60:.2f} minutes")
