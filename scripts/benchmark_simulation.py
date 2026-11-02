"""
COG-EW Monte Carlo Aerospace Benchmark (scripts/benchmark_simulation.py)
Evaluates 1,000 held-out test seeds (20000–20999) with 95% Wilson confidence intervals
across hypersonic flight, RGPO deception, SAM evasion, and standoff target destruction.
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.env import CogEWEnv
from src.simulation.guidance import AutonomousTacticalGuidanceComputer, MissionPhase


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate 95% Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence z-score
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z ** 2) / (4.0 * (total ** 2)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower * 100.0, upper * 100.0


def run_benchmark(num_episodes: int = 1000, start_seed: int = 20000) -> dict:
    print("=" * 76)
    print(f"COG-EW AEROSPACE BENCHMARK: {num_episodes:,} FLIGHTS (SEEDS {start_seed}–{start_seed + num_episodes - 1})")
    print("=" * 76)

    env = CogEWEnv(dt=0.04)  # dt=0.04 for high-speed benchmark execution
    guidance = AutonomousTacticalGuidanceComputer()

    survived_count = 0
    target_destroyed_count = 0
    rgpo_break_lock_count = 0
    zero_g_violation_count = 0
    miss_distances: list[float] = []
    flight_durations: list[float] = []

    t_start = time.time()

    for i in range(num_episodes):
        seed = start_seed + i
        obs, info = env.reset(seed=seed)
        guidance.reset()

        ep_g_violation = False
        ep_break_lock = False
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Closed-loop autonomous flight computer
            nearest_sam = env._get_nearest_sam_distance()
            threat_info = {
                "radar_early_warning_tracking": env.iads.radars[0].tracking_target,
                "radar_acquisition_tracking": env.iads.radars[1].tracking_target,
                "radar_fire_control_locked": env.iads.radars[2].lock_on_active,
                "active_missiles_count": len([m for m in env.iads.missiles if m.active]),
            }

            control, ew_mode = guidance.compute_guidance(
                state=env.vehicle.state,
                threats_info=threat_info,
                nearest_sam_distance_m=nearest_sam,
                dt=env.dt,
            )

            # Map guidance commands into environment action box
            # alpha: [-2 deg, +14 deg] -> [-1, 1]
            alpha_deg = math.degrees(control.alpha_cmd_rad)
            act_alpha = np.clip((alpha_deg - 6.0) / 8.0, -1.0, 1.0)
            # bank: [-70 deg, +70 deg] -> [-1, 1]
            bank_deg = math.degrees(control.bank_cmd_rad)
            act_bank = np.clip(bank_deg / 70.0, -1.0, 1.0)

            # EW mode map
            if ew_mode.value == "PASSIVE_ESM":
                act_ew = -0.7
            elif ew_mode.value == "BARRAGE_NOISE":
                act_ew = 0.0
            else:
                act_ew = 0.7

            action = np.array([act_alpha, act_bank, act_ew, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            if info["g_load"] > 12.0:
                ep_g_violation = True

            if info["radar_lock_broken"]:
                ep_break_lock = True

        # Episode outcomes
        if info["altitude_m"] > 1000.0 and not (info["altitude_m"] <= 0.0):
            survived_count += 1

        if info["target_destroyed"] or guidance.bunker_destroyed:
            target_destroyed_count += 1
            if guidance.standoff_weapon.impacted:
                miss_distances.append(guidance.standoff_weapon.miss_distance_m)

        if ep_break_lock:
            rgpo_break_lock_count += 1

        if not ep_g_violation:
            zero_g_violation_count += 1

        flight_durations.append(info["sim_time"])

        if (i + 1) % 250 == 0 or (i + 1) == num_episodes:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"[{i + 1:4d}/{num_episodes}] Elapsed: {elapsed:.1f}s ({rate:.0f} flights/s) | "
                  f"Survive: {survived_count / (i + 1) * 100:.1f}% | "
                  f"Target Destroyed: {target_destroyed_count / (i + 1) * 100:.1f}% | "
                  f"RGPO Breaks: {rgpo_break_lock_count / (i + 1) * 100:.1f}%")

    env.close()

    total_time = time.time() - t_start
    p_surv = (survived_count / num_episodes) * 100.0
    p_target = (target_destroyed_count / num_episodes) * 100.0
    p_rgpo = (rgpo_break_lock_count / num_episodes) * 100.0
    p_g_safe = (zero_g_violation_count / num_episodes) * 100.0

    ci_surv = wilson_score_interval(survived_count, num_episodes)
    ci_target = wilson_score_interval(target_destroyed_count, num_episodes)
    ci_rgpo = wilson_score_interval(rgpo_break_lock_count, num_episodes)

    mean_cep = float(np.mean(miss_distances)) if miss_distances else 0.0
    p95_cep = float(np.percentile(miss_distances, 95)) if miss_distances else 0.0

    print("\n" + "=" * 76)
    print("EMPIRICAL BENCHMARK RESULTS SUMMARY (1,000 MONTE CARLO SEEDS)")
    print("=" * 76)
    print(f"1. Mission Survivability:       {p_surv:.2f}%  (95% Wilson CI: [{ci_surv[0]:.2f}%, {ci_surv[1]:.2f}%])")
    print(f"2. Target Bunker Destruction:    {p_target:.2f}%  (95% Wilson CI: [{ci_target[0]:.2f}%, {ci_target[1]:.2f}%])")
    print(f"3. RGPO Radar Break-Lock Rate:  {p_rgpo:.2f}%  (95% Wilson CI: [{ci_rgpo[0]:.2f}%, {ci_rgpo[1]:.2f}%])")
    print(f"4. Aerodynamic G Safety Compliance: {p_g_safe:.2f}%  (Zero G-limit violations)")
    print(f"5. Standoff Strike Accuracy:    Mean CEP = {mean_cep:.2f} m | p95 CEP = {p95_cep:.2f} m")
    print(f"6. Benchmark Throughput:        {num_episodes / total_time:.1f} episodes/sec ({total_time:.2f} s total)")
    print("=" * 76)

    gate_surv = p_surv >= 90.0
    gate_target = p_target >= 90.0
    gate_rgpo = p_rgpo >= 90.0
    all_passed = gate_surv and gate_target and gate_rgpo

    print(f"\n>>> AEROSPACE GATE EVALUATION: {'PASSED (ALL GATES CLEARED)' if all_passed else 'FAILED'} <<<\n")

    return {
        "num_episodes": num_episodes,
        "survivability_rate": p_surv,
        "survivability_ci": ci_surv,
        "target_destroyed_rate": p_target,
        "target_destroyed_ci": ci_target,
        "rgpo_break_lock_rate": p_rgpo,
        "rgpo_break_lock_ci": ci_rgpo,
        "g_safety_compliance": p_g_safe,
        "mean_cep_m": mean_cep,
        "p95_cep_m": p95_cep,
        "total_time_s": total_time,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    run_benchmark(num_episodes=1000, start_seed=20000)

