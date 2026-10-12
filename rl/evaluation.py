"""
Evaluation Metrics & Benchmarks (rl/evaluation.py)
Computes key performance metrics across test trajectories:
1. Mission Success Rate (%)
2. Survivability Index
3. Trajectory Quality
4. EW Effectiveness
5. Inference Latency (ms)
"""

import time
import numpy as np


class Evaluator:
    """
    Benchmark suite for evaluating trained EW-Guidance policy models.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run_benchmark(self, num_episodes: int = 20) -> dict:
        successes = 0
        p_detections = []
        latencies_ms = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_p_det = []

            while not done:
                t0 = time.perf_counter()
                action, _, _ = self.agent.select_action(obs, deterministic=True)
                t_latency = (time.perf_counter() - t0) * 1000.0  # ms
                latencies_ms.append(t_latency)

                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_p_det.append(info.get("max_p_detection", 0.0))

                if terminated and info.get("status") == "target_reached":
                    successes += 1

                done = terminated or truncated

            p_detections.append(np.mean(ep_p_det))

        metrics = {
            "mission_success_rate": (successes / num_episodes) * 100.0,
            "survivability_index": 1.0 - float(np.mean(p_detections)),
            "mean_inference_latency_ms": float(np.mean(latencies_ms)),
            "max_inference_latency_ms": float(np.max(latencies_ms)),
            "latency_constraint_met": float(np.max(latencies_ms)) < 50.0
        }
        return metrics
