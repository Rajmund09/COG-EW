from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


def get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"[PASS] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the COG-EW dashboard API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--index", type=int, default=5000)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print("\nCOG-EW Dashboard Verification")
    print("=" * 38)

    try:
        # 1. Backend health
        health = get_json(f"{base_url}/api/health")
        check(health.get("status") == "ok", "API health endpoint responds")
        check(
            health.get("model_exists") is True,
            "Trained CNN checkpoint exists",
        )
        check(
            health.get("test_dataset_exists") is True,
            "Held-out test dataset exists",
        )
        print("Device:", health.get("device"))
        print("GPU:", health.get("gpu"))

        # 2. Evaluation metrics
        metrics = get_json(f"{base_url}/api/metrics")
        check(isinstance(metrics, dict), "Evaluation metrics are valid JSON")
        print("Metric report keys:", ", ".join(metrics.keys()) or "(none)")

        # 3. Single-sample prediction
        prediction = post_json(
            f"{base_url}/api/predict",
            {"test_index": args.index},
        )

        classes = ["8PSK", "BPSK", "QPSK", "16QAM", "64QAM"]
        predicted = prediction.get("predicted_class")
        probabilities = prediction.get("probabilities")

        check(
            predicted in classes,
            "Prediction contains a recognized modulation class",
        )
        check(
            isinstance(probabilities, dict)
            and set(probabilities.keys()) == set(classes),
            "All five class probabilities are present",
        )

        values = [float(probabilities[name]) for name in classes]
        check(
            all(math.isfinite(value) and 0 <= value <= 1 for value in values),
            "Probabilities are finite and between 0 and 1",
        )
        check(
            abs(sum(values) - 1.0) < 1e-4,
            "Class probabilities sum to approximately 1",
        )

        print("\nPrediction details")
        print("-" * 38)
        print("Test index:", prediction.get("test_index"))
        print("True class:", prediction.get("true_class"))
        print("Predicted class:", predicted)
        print("Confidence:", f"{prediction.get('confidence', 0):.2%}")
        print("Dataset SNR:", prediction.get("snr_db"), "dB")
        print("Dataset source:", prediction.get("source"))
        print("Inference time:", prediction.get("inference_ms"), "ms")
        print("Correct:", prediction.get("correct"))

        print("\nClass probabilities")
        for name in classes:
            print(f"  {name:>5}: {probabilities[name]:.2%}")

        print("\nRESULT: API verification completed successfully.")
        return 0

    except urllib.error.HTTPError as exc:
        print(f"\n[FAIL] HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        try:
            print(exc.read().decode("utf-8"), file=sys.stderr)
        except Exception:
            pass
        return 1

    except urllib.error.URLError as exc:
        print(
            "\n[FAIL] Cannot connect to the API. " "Start the FastAPI server first.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1

    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
