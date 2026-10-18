"""Evaluation metrics and offline benchmark validation."""

import time
from collections.abc import Mapping

import numpy as np


REQUIRED_METRICS = (
    "mission_success_rate",
    "survivability_index",
    "mean_inference_latency_ms",
    "max_inference_latency_ms",
    "latency_constraint_met",
)

LATENCY_LIMIT_MS = 50.0


def validate_metrics(metrics: Mapping) -> dict:
    """Validate and normalize benchmark metrics before reporting them."""
    if not isinstance(metrics, Mapping):
        raise TypeError("Evaluation metrics must be a mapping.")

    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"Missing evaluation metrics: {missing}")

    numeric_ranges = {
        "mission_success_rate": (0.0, 100.0),
        "survivability_index": (0.0, 1.0),
        "mean_inference_latency_ms": (0.0, float("inf")),
        "max_inference_latency_ms": (0.0, float("inf")),
    }

    normalized = {}

    for key, (minimum, maximum) in numeric_ranges.items():
        value = metrics[key]

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be numeric, not boolean.")

        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a finite number.") from exc

        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite.")

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{key} must be in [{minimum}, {maximum}]; received {value}."
            )

        normalized[key] = value

    if normalized["max_inference_latency_ms"] < normalized["mean_inference_latency_ms"]:
        raise ValueError("Maximum inference latency cannot be below mean latency.")

    latency_flag = metrics["latency_constraint_met"]
    if not isinstance(latency_flag, (bool, np.bool_)):
        raise ValueError("latency_constraint_met must be a boolean.")

    expected_latency_flag = normalized["max_inference_latency_ms"] < LATENCY_LIMIT_MS
    if bool(latency_flag) != expected_latency_flag:
        raise ValueError(
            "latency_constraint_met conflicts with maximum latency "
            f"and the {LATENCY_LIMIT_MS:.0f} ms limit."
        )

    normalized["latency_constraint_met"] = bool(latency_flag)
    return normalized


class Evaluator:
    """Benchmark runner for offline policy evaluation."""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run_benchmark(self, num_episodes: int = 20) -> dict:
        if (
            not isinstance(num_episodes, int)
            or isinstance(num_episodes, bool)
            or num_episodes < 1
        ):
            raise ValueError("num_episodes must be a positive integer.")

        successes = 0
        episode_detection_means = []
        latencies_ms = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_detections = []

            while not done:
                start = time.perf_counter()
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                latency_ms = (time.perf_counter() - start) * 1000.0

                if not np.isfinite(latency_ms) or latency_ms < 0:
                    raise ValueError("Inference latency is invalid.")

                latencies_ms.append(float(latency_ms))

                obs, _, terminated, truncated, info = self.env.step(action)

                if not isinstance(info, Mapping):
                    raise ValueError("Environment info must be a mapping.")

                if "max_p_detection" not in info:
                    raise ValueError("Environment info is missing 'max_p_detection'.")

                try:
                    p_detection = float(info["max_p_detection"])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Detection probability must be numeric.") from exc

                if not np.isfinite(p_detection) or not 0.0 <= p_detection <= 1.0:
                    raise ValueError(
                        "Detection probability must be finite and in [0, 1]."
                    )

                episode_detections.append(p_detection)

                if terminated and info.get("status") == "target_reached":
                    successes += 1

                done = bool(terminated or truncated)

            if not episode_detections:
                raise ValueError(
                    "An evaluation episode produced no detection observations."
                )

            episode_detection_means.append(float(np.mean(episode_detections)))

        if not latencies_ms:
            raise ValueError("Evaluation produced no inference latency samples.")

        max_latency = float(np.max(latencies_ms))

        metrics = {
            "mission_success_rate": (successes / num_episodes) * 100.0,
            "survivability_index": (1.0 - float(np.mean(episode_detection_means))),
            "mean_inference_latency_ms": float(np.mean(latencies_ms)),
            "max_inference_latency_ms": max_latency,
            "latency_constraint_met": max_latency < LATENCY_LIMIT_MS,
        }

        return validate_metrics(metrics)
