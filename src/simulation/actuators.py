"""
Aerospace Actuator Suite
Models aerodynamic elevons, Reaction Control System (RCS) gas nozzles,
engine throttle response, and jammer gimbal slew dynamics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class ActuatorState:
    elevon_deflection_deg: float = 0.0
    bank_deflection_deg: float = 0.0
    throttle_actual: float = 0.0
    rcs_firing: bool = False
    jammer_azimuth_deg: float = 0.0
    jammer_elevation_deg: float = 0.0


class ActuatorSuite:
    """
    Simulates physical lag, maximum deflections, and slew rate limits.
    """

    def __init__(
        self,
        max_elevon_rate_deg_s: float = 30.0,
        max_elevon_deg: float = 20.0,
        max_bank_rate_deg_s: float = 45.0,
        max_bank_deg: float = 75.0,
        rcs_thrust_n: float = 450.0,           # Stratospheric cold gas thruster force
        rcs_moment_arm_m: float = 3.5,
    ):
        self.max_elevon_rate = max_elevon_rate_deg_s
        self.max_elevon = max_elevon_deg
        self.max_bank_rate = max_bank_rate_deg_s
        self.max_bank = max_bank_deg
        self.rcs_thrust = rcs_thrust_n
        self.rcs_arm = rcs_moment_arm_m

        self.state = ActuatorState()

    def update(
        self,
        cmd_alpha_rad: float,
        cmd_bank_rad: float,
        cmd_throttle: float,
        dt: float = 0.02,
        altitude_m: float = 29500.0,
        dynamic_pressure_pa: float = 22000.0,
    ) -> tuple[float, float, float, float]:
        """
        Step actuator states toward commands with rate saturation.
        Returns: (actual_alpha_rad, actual_bank_rad, actual_throttle, rcs_yaw_torque)
        """
        cmd_elevon_deg = math.degrees(cmd_alpha_rad)
        cmd_bank_deg = math.degrees(cmd_bank_rad)

        # Rate-limited elevon update
        delta_elevon = cmd_elevon_deg - self.state.elevon_deflection_deg
        max_step_elevon = self.max_elevon_rate * dt
        delta_elevon_clamped = np.clip(delta_elevon, -max_step_elevon, max_step_elevon)
        self.state.elevon_deflection_deg = float(np.clip(
            self.state.elevon_deflection_deg + delta_elevon_clamped,
            -self.max_elevon, self.max_elevon
        ))

        # Rate-limited bank angle update
        delta_bank = cmd_bank_deg - self.state.bank_deflection_deg
        max_step_bank = self.max_bank_rate * dt
        delta_bank_clamped = np.clip(delta_bank, -max_step_bank, max_step_bank)
        self.state.bank_deflection_deg = float(np.clip(
            self.state.bank_deflection_deg + delta_bank_clamped,
            -self.max_bank, self.max_bank
        ))

        # First-order throttle lag (tau = 0.25 s)
        tau_throttle = 0.25
        self.state.throttle_actual += (cmd_throttle - self.state.throttle_actual) * (dt / tau_throttle)
        self.state.throttle_actual = float(np.clip(self.state.throttle_actual, 0.0, 1.0))

        # RCS Reaction Control System active when in thin air (q < 6 kPa)
        rcs_torque = 0.0
        if dynamic_pressure_pa < 6000.0 and abs(delta_bank) > 5.0:
            self.state.rcs_firing = True
            rcs_sign = 1.0 if delta_bank > 0 else -1.0
            rcs_torque = rcs_sign * self.rcs_thrust * self.rcs_arm
        else:
            self.state.rcs_firing = False

        return (
            math.radians(self.state.elevon_deflection_deg),
            math.radians(self.state.bank_deflection_deg),
            self.state.throttle_actual,
            rcs_torque,
        )

