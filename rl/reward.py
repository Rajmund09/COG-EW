"""
Multi-Objective Reward Engine (rl/reward.py)
Calculates Pareto trade-off rewards for mission success, survivability, heat, fuel, and time.
"""

import numpy as np


class MultiObjectiveRewardEngine:
    """
    Multi-objective reward calculator:
    r_total = w1*r_mission + w2*r_survivability + w3*r_thermal + w4*r_energy + w5*r_time
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "mission": 1.0,
            "survivability": 2.0,
            "thermal": 0.5,
            "energy": 0.2,
            "time": 0.1
        }

    def compute_reward(
        self,
        dist_to_target: float,
        p_detection_max: float,
        thermal_heat: float,
        jammer_power_kw: float,
        step_count: int
    ) -> float:
        r_mission = -dist_to_target / 50000.0
        r_surv = -np.exp(3.0 * p_detection_max) + 1.0  # Non-linear penalty for high detection
        r_therm = -max(0.0, thermal_heat - 1000.0) / 500.0
        r_energy = -jammer_power_kw / 10.0
        r_time = -0.01 * step_count

        total_reward = (
            self.weights["mission"] * r_mission +
            self.weights["survivability"] * r_surv +
            self.weights["thermal"] * r_therm +
            self.weights["energy"] * r_energy +
            self.weights["time"] * r_time
        )
        return float(total_reward)
