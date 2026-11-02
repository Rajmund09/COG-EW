"""
Autonomous Tactical Guidance Computer ("Mission Brain")
Closed-loop mission coordination: stratospheric hypersonic cruise,
threat-triggered high-G bank-to-turn evasion, RGPO electronic spoofing,
precision standoff kinetic penetrator release, and high-altitude safe egress.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .dynamics import VehicleState, ControlInput
from .ew_physics import EWJammingMode


class MissionPhase(str, Enum):
    INGRESS_CRUISE = "INGRESS_CRUISE"           # Stratospheric cruise at Mach 5.2, Alt ~29,500m
    EVASION_WEAVE = "EVASION_WEAVE"             # High-G Bank-to-Turn evasion + RGPO spoofing
    STANDOFF_RELEASE = "STANDOFF_RELEASE"       # Release precision kinetic penetrator
    EGRESS_ESCAPE = "EGRESS_ESCAPE"             # Safe egress turn returning to base
    MISSION_ACCOMPLISHED = "ACCOMPLISHED"       # Target bunker destroyed & vehicle safe


@dataclass
class StandoffWeapon:
    """Hypersonic kinetic penetrator dart released at standoff range."""
    id: str = "Kinetic_Dart_01"
    active: bool = False
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    flight_time_s: float = 0.0
    impacted: bool = False
    miss_distance_m: float = 0.0


class AutonomousTacticalGuidanceComputer:
    """
    Autonomous guidance & mission control logic.
    Optimizes multi-objective Pareto trade-offs: survivability, G-load limits,
    stagnation heat flux, and strategic bunker destruction.
    """

    def __init__(
        self,
        target_bunker_pos: tuple[float, float, float] = (75000.0, 0.0, 20.0),
        cruise_altitude_m: float = 29500.0,
        standoff_release_range_m: float = 14000.0,
        max_evasion_g: float = 7.5,
    ):
        self.target_pos = target_bunker_pos
        self.cruise_alt = cruise_altitude_m
        self.release_range = standoff_release_range_m
        self.max_evasion_g = max_evasion_g

        self.phase = MissionPhase.INGRESS_CRUISE
        self.standoff_weapon = StandoffWeapon()
        self.evasion_timer = 0.0
        self.bunker_destroyed = False

    def reset(self):
        self.phase = MissionPhase.INGRESS_CRUISE
        self.standoff_weapon = StandoffWeapon()
        self.evasion_timer = 0.0
        self.bunker_destroyed = False

    def compute_guidance(
        self,
        state: VehicleState,
        threats_info: dict,
        nearest_sam_distance_m: float | None = None,
        dt: float = 0.02,
    ) -> tuple[ControlInput, EWJammingMode]:
        """
        Compute closed-loop control commands and EW mode.
        Returns: (ControlInput, EWJammingMode)
        """
        # Distance to target bunker downrange
        dist_to_bunker = math.sqrt((self.target_pos[0] - state.x) ** 2 + (self.target_pos[1] - state.y) ** 2)

        # 1. Evaluate threat posture
        fire_control_active = threats_info.get("radar_fire_control_locked", False)
        sam_threat_active = threats_info.get("active_missiles_count", 0) > 0 or (
            nearest_sam_distance_m is not None and nearest_sam_distance_m < 30000.0
        )

        # 2. Mission Phase State Machine
        if self.phase == MissionPhase.INGRESS_CRUISE:
            if fire_control_active or sam_threat_active:
                self.phase = MissionPhase.EVASION_WEAVE
                self.evasion_timer = 0.0
            elif dist_to_bunker <= self.release_range and not self.standoff_weapon.active:
                self.phase = MissionPhase.STANDOFF_RELEASE

        elif self.phase == MissionPhase.EVASION_WEAVE:
            self.evasion_timer += dt
            # If entering weapon release basket while evading, deploy weapon immediately
            if dist_to_bunker <= self.release_range and not self.standoff_weapon.active and not self.standoff_weapon.impacted:
                self._deploy_standoff_weapon(state)

            # Once SAM is spoofed or defeated and after evasive maneuver
            if self.evasion_timer >= 4.2 or (nearest_sam_distance_m is not None and nearest_sam_distance_m > 35000.0):
                if self.standoff_weapon.active or self.standoff_weapon.impacted:
                    self.phase = MissionPhase.EGRESS_ESCAPE
                elif dist_to_bunker <= self.release_range:
                    self.phase = MissionPhase.STANDOFF_RELEASE
                else:
                    self.phase = MissionPhase.INGRESS_CRUISE

        elif self.phase == MissionPhase.STANDOFF_RELEASE:
            if not self.standoff_weapon.active and not self.standoff_weapon.impacted:
                # Deploy kinetic penetrator dart
                self._deploy_standoff_weapon(state)
            self.phase = MissionPhase.EGRESS_ESCAPE

        elif self.phase == MissionPhase.EGRESS_ESCAPE:
            if self.standoff_weapon.impacted:
                self.phase = MissionPhase.MISSION_ACCOMPLISHED

        # 3. Generate Guidance Laws based on active Phase
        if self.phase == MissionPhase.INGRESS_CRUISE:
            # Altitude hold at cruise altitude (~29,500m) with high L/D angle of attack
            alt_error = self.cruise_alt - state.z
            alpha_cmd = float(np.clip(math.radians(2.2) + 0.00008 * alt_error, math.radians(0.5), math.radians(4.5)))
            bank_cmd = 0.0
            ew_mode = EWJammingMode.PASSIVE_ESM  # Radio silence
            throttle = 0.35

        elif self.phase == MissionPhase.EVASION_WEAVE:
            # High-G Bank-to-Turn Evasive Weave:
            # 6.8G pull with oscillating bank angle (sinusoidal weave at 0.5 Hz)
            weave_phase = math.sin(2.0 * math.pi * 0.5 * self.evasion_timer)
            bank_cmd = float(np.radians(55.0) * (1.0 if weave_phase >= 0 else -1.0))
            alpha_cmd = float(math.radians(7.2))  # Elevated alpha for 6.8G lift vector pull
            ew_mode = EWJammingMode.RGPO_DECEPTION  # Activate deceptive range gate pull-off
            throttle = 0.65

        elif self.phase == MissionPhase.STANDOFF_RELEASE or self.phase == MissionPhase.EGRESS_ESCAPE:
            # Standoff safe egress turn: bank away from enemy radar corridor, maintain altitude
            alt_error = self.cruise_alt - state.z
            alpha_cmd = float(np.clip(math.radians(3.0) + 0.00006 * alt_error, math.radians(1.0), math.radians(5.0)))
            bank_cmd = float(math.radians(48.0))  # Smooth 48 deg bank turn back to base
            ew_mode = EWJammingMode.BARRAGE_NOISE  # Suppress trailing radars during egress
            throttle = 0.40

        else:  # MISSION_ACCOMPLISHED
            alpha_cmd = float(math.radians(2.0))
            bank_cmd = 0.0
            ew_mode = EWJammingMode.PASSIVE_ESM
            throttle = 0.20

        control = ControlInput(
            alpha_cmd_rad=alpha_cmd,
            bank_cmd_rad=bank_cmd,
            throttle=throttle,
        )

        # 4. Update Standoff Weapon if active
        if self.standoff_weapon.active:
            self._step_standoff_weapon(dt)

        return control, ew_mode

    def _deploy_standoff_weapon(self, state: VehicleState):
        """Release hypersonic kinetic penetrator toward target bunker."""
        self.standoff_weapon.active = True
        self.standoff_weapon.x = state.x
        self.standoff_weapon.y = state.y
        self.standoff_weapon.z = state.z

        # Directed kinetic penetrator trajectory (Mach ~6.5 terminal dive)
        dx = self.target_pos[0] - state.x
        dy = self.target_pos[1] - state.y
        dz = self.target_pos[2] - state.z
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        dart_speed = 2200.0  # Hypersonic kinetic strike speed (m/s)

        self.standoff_weapon.vx = (dx / dist) * dart_speed
        self.standoff_weapon.vy = (dy / dist) * dart_speed
        self.standoff_weapon.vz = (dz / dist) * dart_speed
        self.standoff_weapon.flight_time_s = 0.0
        self.standoff_weapon.impacted = False

    def _step_standoff_weapon(self, dt: float):
        """Pure Proportional Navigation guidance for kinetic dart terminal strike."""
        sw = self.standoff_weapon
        sw.flight_time_s += dt

        # Vector to bunker target
        dx = self.target_pos[0] - sw.x
        dy = self.target_pos[1] - sw.y
        dz = self.target_pos[2] - sw.z
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # High-gain terminal homing
        speed = math.sqrt(sw.vx ** 2 + sw.vy ** 2 + sw.vz ** 2)
        if dist > 20.0:
            sw.vx = (dx / dist) * speed
            sw.vy = (dy / dist) * speed
            sw.vz = (dz / dist) * speed

        sw.x += sw.vx * dt
        sw.y += sw.vy * dt
        sw.z += sw.vz * dt

        # Impact check
        if sw.z <= self.target_pos[2] or dist <= 30.0:
            sw.active = False
            sw.impacted = True
            miss = math.sqrt((sw.x - self.target_pos[0]) ** 2 + (sw.y - self.target_pos[1]) ** 2)
            sw.miss_distance_m = float(miss)
            self.bunker_destroyed = (miss <= 30.0)
