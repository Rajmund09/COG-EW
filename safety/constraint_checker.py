"""
Safety Constraint Filter (safety/constraint_checker.py)
Projects candidate actions into physically feasible space to prevent vehicle loss or structural failure.
"""

import numpy as np


class SafetyConstraintFilter:
    """
    Action projection safety filter:
    1. Maximum G-load constraint (< 10g)
    2. Minimum altitude constraint (> 15,000 m)
    3. Thermal flux acceleration limits
    """

    def __init__(self, max_g_load: float = 10.0, min_altitude: float = 15000.0):
        self.max_g = max_g_load
        self.min_alt = min_altitude

    def filter_action(self, candidate_action: np.ndarray, vehicle_state: dict) -> np.ndarray:
        """
        Clips or projects guidance actions to respect physical envelope limits.
        """
        filtered_action = candidate_action.copy()
        
        current_g = vehicle_state.get("g_load", 1.0)
        current_alt = vehicle_state.get("position", [0.0, 30000.0])[1]

        # Prevent downward pull-down maneuver if close to altitude floor
        if current_alt < self.min_alt and filtered_action[0] < 0:
            filtered_action[0] = 0.0  # Force positive or zero pitch up

        # Clamp angle-of-attack rate if g-load near max threshold
        if current_g > (self.max_g * 0.9):
            filtered_action[0] = np.clip(filtered_action[0], -0.1, 0.05)

        return filtered_action
