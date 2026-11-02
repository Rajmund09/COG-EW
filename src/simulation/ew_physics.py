"""
First-Principles Cognitive Electronic Warfare (EW) Physics
Implements the Radar Range Equation, Jammer-to-Signal (J/S) power ratio,
Barrage Noise Jamming noise-floor elevation, and continuous Range Gate Pull-Off (RGPO) walk-off.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum


class EWJammingMode(str, Enum):
    PASSIVE_ESM = "PASSIVE_ESM"              # Radio silence, passive intercept
    BARRAGE_NOISE = "BARRAGE_NOISE"          # Broadband RF noise saturation
    RGPO_DECEPTION = "RGPO_DECEPTION"        # Range Gate Pull-Off deception spoofing


@dataclass
class EWState:
    mode: EWJammingMode = EWJammingMode.PASSIVE_ESM
    jammer_power_kw: float = 45.0             # Jammer peak ERP (kW)
    rgpo_active: bool = False
    rgpo_phase: str = "IDLE"                  # IDLE -> CAPTURE -> PULL_OFF -> BREAK_LOCK
    rgpo_delay_ns: float = 0.0                # Added pulse time delay (ns)
    rgpo_apparent_range_offset_m: float = 0.0 # Apparent false phantom distance offset (m)
    current_js_db: float = 0.0                # Current Jammer-to-Signal ratio (dB)
    radar_lock_broken: bool = False
    radar_noise_elevation_db: float = 0.0


class CognitiveEWPhysics:
    """
    Computes rigorous radar cross-section (RCS) reflections,
    received powers, Jammer-to-Signal ratios, and RGPO track-gate pull-off.
    """

    SPEED_OF_LIGHT = 299792458.0  # m/s

    def __init__(
        self,
        base_rcs_m2: float = 0.045,            # Waverider frontal radar cross section (m^2)
        jammer_max_power_kw: float = 50.0,     # Jammer ERP (kW)
        rgpo_pull_rate_m_s2: float = 160.0,    # Pull-off acceleration (m/s^2)
        rgpo_gate_width_m: float = 120.0,      # Radar tracker range gate width (m)
    ):
        self.base_rcs = base_rcs_m2
        self.max_jammer_kw = jammer_max_power_kw
        self.rgpo_pull_rate = rgpo_pull_rate_m_s2
        self.rgpo_gate_width = rgpo_gate_width_m

        self.state = EWState()
        self._rgpo_timer = 0.0

    def compute_rcs(self, aspect_angle_rad: float) -> float:
        """
        Aspect-dependent Radar Cross Section (RCS).
        Frontal aspect (~0 rad) is low observable (0.045 m^2).
        Broadside aspect (~pi/2 rad) reflects higher RCS (~1.2 m^2).
        """
        abs_aspect = abs(aspect_angle_rad)
        sin_factor = math.sin(abs_aspect) ** 2
        return self.base_rcs + 1.15 * sin_factor

    def compute_js_ratio(
        self,
        distance_m: float,
        radar_peak_power_kw: float,
        radar_gain_db: float,
        aspect_angle_rad: float = 0.0,
    ) -> float:
        """
        Compute Jammer-to-Signal ratio (J/S) in dB:
        J/S = (P_j * G_j / (P_t * G_t)) * (4 * pi * R^2 / sigma)
        """
        dist = max(100.0, distance_m)
        rcs = self.compute_rcs(aspect_angle_rad)

        g_t = math.pow(10.0, radar_gain_db / 10.0)
        p_t = radar_peak_power_kw * 1000.0
        p_j_erp = self.state.jammer_power_kw * 1000.0

        # Radar Skin Echo Power received: P_r = (P_t * G_t^2 * lambda^2 * sigma) / ((4*pi)^3 * R^4)
        # Jammer Power received at radar: P_rj = (P_j * G_t * lambda^2) / ((4*pi)^2 * R^2)
        # Ratio J/S = (P_j / P_t) * (1 / G_t) * (4 * pi * R^2 / sigma)
        js_linear = (p_j_erp / p_t) * (1.0 / g_t) * (4.0 * math.pi * (dist ** 2) / max(0.001, rcs))
        js_db = 10.0 * math.log10(max(1e-6, js_linear))
        self.state.current_js_db = js_db
        return js_db

    def step(
        self,
        mode: EWJammingMode,
        distance_to_radar_m: float,
        radar_locked: bool,
        dt: float = 0.02,
    ) -> EWState:
        """Step the EW physics state machine across time step dt."""
        self.state.mode = mode

        if mode == EWJammingMode.PASSIVE_ESM:
            self.state.rgpo_active = False
            self.state.rgpo_phase = "IDLE"
            self.state.rgpo_apparent_range_offset_m = 0.0
            self.state.radar_noise_elevation_db = 0.0
            self._rgpo_timer = 0.0
            return self.state

        elif mode == EWJammingMode.BARRAGE_NOISE:
            self.state.rgpo_active = False
            self.state.rgpo_phase = "NOISE_SATURATION"
            # Elevate noise floor inversely proportional to R^2
            dist = max(500.0, distance_to_radar_m)
            self.state.radar_noise_elevation_db = float(min(28.0, 45.0 * (15000.0 / dist) ** 2))
            return self.state

        elif mode == EWJammingMode.RGPO_DECEPTION:
            self.state.rgpo_active = True
            self.state.radar_noise_elevation_db = 0.0

            if not radar_locked:
                self.state.rgpo_phase = "IDLE"
                self._rgpo_timer = 0.0
                return self.state

            # RGPO Deception Sequence
            self._rgpo_timer += dt

            if self._rgpo_timer < 0.6:
                # Phase 1: Set-on & Capture. Synchronize pulse with high power J/S > 12 dB
                self.state.rgpo_phase = "CAPTURE"
                self.state.rgpo_apparent_range_offset_m = 0.0
                self.state.radar_lock_broken = False

            elif self._rgpo_timer < 3.2:
                # Phase 2: Pull-Off Walk. Ramp delay creating accelerating phantom distance
                self.state.rgpo_phase = "PULL_OFF"
                t_pull = self._rgpo_timer - 0.6
                offset = 0.5 * self.rgpo_pull_rate * (t_pull ** 2)
                self.state.rgpo_apparent_range_offset_m = float(offset)
                self.state.rgpo_delay_ns = (2.0 * offset / self.SPEED_OF_LIGHT) * 1e9

                # Check if false echo has pulled out of radar range gate
                if offset > self.rgpo_gate_width:
                    self.state.rgpo_phase = "BREAK_LOCK"
                    self.state.radar_lock_broken = True

            else:
                # Phase 3: Gate drop & break-lock achieved
                self.state.rgpo_phase = "BREAK_LOCK"
                self.state.radar_lock_broken = True

            return self.state

        return self.state

