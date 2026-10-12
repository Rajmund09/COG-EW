"""
Baseline Guidance Tracker (guidance/baseline_guidance.py)
Implements non-AI baseline guidance laws including Proportional Navigation (PNG).
"""

import numpy as np


class ProportionalNavigation:
    """
    Standard 2D Pure Proportional Navigation Guidance (PNG) tracker.
    Formula: a_cmd = N * V_c * d(LOS)/dt
    """

    def __init__(self, navigation_gain: float = 3.5):
        self.N = navigation_gain
        self.prev_los_angle = None

    def reset(self):
        self.prev_los_angle = None

    def compute_guidance_command(
        self,
        vehicle_pos: np.ndarray,
        vehicle_vel: float,
        heading: float,
        target_pos: np.ndarray,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Calculates commanded angle-of-attack and bank angle acceleration to intercept target.
        """
        rel_pos = target_pos - vehicle_pos
        los_angle = np.arctan2(rel_pos[1], rel_pos[0])

        if self.prev_los_angle is None:
            los_rate = 0.0
        else:
            # Handle angle wrapping
            los_diff = los_angle - self.prev_los_angle
            los_rate = np.arctan2(np.sin(los_diff), np.cos(los_diff)) / dt

        self.prev_los_angle = los_angle

        # Closing velocity approximation
        v_closing = vehicle_vel

        # Commanded normal acceleration (m/s^2)
        accel_cmd = self.N * v_closing * los_rate

        # Convert acceleration command to angle of attack delta
        alpha_cmd = np.clip(accel_cmd / 50.0, np.radians(-15), np.radians(20))
        bank_cmd = 0.0  # Planar guidance

        return np.array([alpha_cmd, bank_cmd], dtype=np.float32)
