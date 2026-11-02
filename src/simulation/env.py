"""
Gymnasium Continuous Environment for COG-EW (CogEWEnv)
Complies with Gymnasium API (v1.0+) featuring continuous observation & action spaces,
Runge-Kutta 4th Order continuous aerospace dynamics, and multi-objective Pareto reward.
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .atmosphere import USStandardAtmosphere1976
from .aerodynamics import HypersonicWaveriderAerodynamics
from .dynamics import ContinuousHypersonicVehicle, VehicleState, ControlInput, RK4Integrator
from .actuators import ActuatorSuite
from .ew_physics import CognitiveEWPhysics, EWJammingMode
from .threats import MultiStaticIADSNetwork
from .guidance import AutonomousTacticalGuidanceComputer, MissionPhase


class CogEWEnv(gym.Env):
    """
    Continuous hypersonic electronic warfare and guidance environment.

    State Observation Space (Box of shape (28,)):
      - Vehicle kinematics [0:6]: [x/100km, y/10km, z/50km, vx/2000m/s, vy/500m/s, vz/500m/s]
      - Aerodynamic attitude & rates [6:12]: [roll, pitch, yaw, p, q, r]
      - Consumables & thermal [12:15]: [fuel_remaining_ratio, g_load/12G, heat_flux_norm]
      - Radar emitter states [15:18]: [VHF_tracking, S_Band_tracking, X_Band_locked]
      - Nearest SAM missile [18:22]: [sam_active, rel_dist/50km, closing_velocity/2000m/s, los_rate]
      - EW Electronic Warfare state [22:25]: [js_ratio_db/30dB, rgpo_phase_id, lock_broken_flag]
      - Target bunker geometry [25:28]: [dist_to_target/100km, bearing_to_target, target_destroyed]

    Continuous Action Space (Box of shape (4,)):
      - action[0]: delta_alpha command in [-1.0, 1.0] -> maps to [-5.0 deg, +15.0 deg]
      - action[1]: bank command in [-1.0, 1.0] -> maps to [-75.0 deg, +75.0 deg]
      - action[2]: jammer mode selector [-1.0, 1.0] -> [-1, -0.33): Passive, [-0.33, 0.33): Noise, [0.33, 1]: RGPO
      - action[3]: standoff weapon release trigger > 0.5
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, dt: float = 0.02):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt

        # Observation space: 28 continuous physical metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )

        # Action space: 4 continuous actuators
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Core subsystems
        self.vehicle = ContinuousHypersonicVehicle()
        self.actuators = ActuatorSuite()
        self.ew = CognitiveEWPhysics()
        self.iads = MultiStaticIADSNetwork()
        self.guidance = AutonomousTacticalGuidanceComputer()

        self.sim_time = 0.0
        self.steps_count = 0
        self.max_steps = 3000  # 60 seconds at dt=0.02
        self.target_destroyed = False
        self.mission_success = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.sim_time = 0.0
        self.steps_count = 0
        self.target_destroyed = False
        self.mission_success = False

        # Randomize initial perturbation if seed provided
        init_alt = 29500.0
        init_mach = 5.2
        if seed is not None:
            rng = np.random.default_rng(seed)
            init_alt += float(rng.uniform(-500.0, 500.0))
            init_mach += float(rng.uniform(-0.15, 0.15))

        sound_speed = USStandardAtmosphere1976.speed_of_sound(init_alt)
        init_vx = init_mach * sound_speed

        self.vehicle.state = VehicleState(
            x=0.0, y=0.0, z=init_alt,
            vx=init_vx, vy=0.0, vz=0.0,
            mass_fuel=self.vehicle.initial_fuel,
        )

        self.actuators = ActuatorSuite()
        self.ew = CognitiveEWPhysics()
        self.iads.reset()
        self.guidance.reset()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps_count += 1
        self.sim_time += self.dt

        action = np.clip(action, -1.0, 1.0)

        # 1. Decode actions
        # Alpha cmd: [-1, 1] -> [-2 deg, +14 deg]
        alpha_cmd = math.radians(6.0 + 8.0 * float(action[0]))
        # Bank cmd: [-1, 1] -> [-70 deg, +70 deg]
        bank_cmd = math.radians(70.0 * float(action[1]))

        # Jammer mode
        jam_val = float(action[2])
        if jam_val < -0.33:
            chosen_ew_mode = EWJammingMode.PASSIVE_ESM
        elif jam_val < 0.33:
            chosen_ew_mode = EWJammingMode.BARRAGE_NOISE
        else:
            chosen_ew_mode = EWJammingMode.RGPO_DECEPTION

        # 2. Update physical actuators
        atm = USStandardAtmosphere1976.get_state(self.vehicle.state.z)
        q = 0.5 * atm.density_kg_m3 * (self.vehicle.state.airspeed ** 2)

        actual_alpha, actual_bank, actual_throttle, rcs_torque = self.actuators.update(
            cmd_alpha_rad=alpha_cmd,
            cmd_bank_rad=bank_cmd,
            cmd_throttle=0.35,
            dt=self.dt,
            altitude_m=self.vehicle.state.z,
            dynamic_pressure_pa=q,
        )

        control = ControlInput(
            alpha_cmd_rad=actual_alpha,
            bank_cmd_rad=actual_bank,
            throttle=actual_throttle,
            rcs_yaw_torque=rcs_torque,
        )

        # 3. Explicit RK4 numerical integration of vehicle kinematics
        self.vehicle.state = RK4Integrator.step(
            vehicle=self.vehicle,
            state=self.vehicle.state,
            control=control,
            dt=self.dt,
        )

        # 4. Electronic warfare physics step
        fc_radar = self.iads.radars[2]
        dist_to_fc = math.sqrt(
            (fc_radar.x - self.vehicle.state.x) ** 2 +
            (fc_radar.y - self.vehicle.state.y) ** 2 +
            (fc_radar.z - self.vehicle.state.z) ** 2
        )
        self.ew.compute_js_ratio(
            distance_m=dist_to_fc,
            radar_peak_power_kw=fc_radar.peak_power_kw,
            radar_gain_db=fc_radar.antenna_gain_db,
        )
        ew_state = self.ew.step(
            mode=chosen_ew_mode,
            distance_to_radar_m=dist_to_fc,
            radar_locked=fc_radar.lock_on_active,
            dt=self.dt,
        )

        # Calculate RGPO phantom coordinates if active
        rgpo_phantom = None
        if ew_state.rgpo_active and ew_state.rgpo_apparent_range_offset_m > 0:
            offset = ew_state.rgpo_apparent_range_offset_m
            rgpo_phantom = (
                self.vehicle.state.x - offset * 0.85,
                self.vehicle.state.y - offset * 0.45,
                self.vehicle.state.z + offset * 0.15,
            )

        # Check SAM launch trigger: when Fire-Control radar locks on and no active SAM
        if fc_radar.lock_on_active and len([m for m in self.iads.missiles if m.active]) == 0:
            self.iads.launch_sam((self.vehicle.state.x, self.vehicle.state.y, self.vehicle.state.z))

        # 5. Step IADS threats & SAM missiles
        threat_status = self.iads.step_threats(
            vehicle_pos=(self.vehicle.state.x, self.vehicle.state.y, self.vehicle.state.z),
            rgpo_phantom_pos=rgpo_phantom,
            rgpo_lock_broken=ew_state.radar_lock_broken,
            noise_elevation_db=ew_state.radar_noise_elevation_db,
            dt=self.dt,
        )

        # 6. Step Standoff Weapon / Autonomous Guidance check
        nearest_sam_dist = self._get_nearest_sam_distance()
        _, auto_ew_mode = self.guidance.compute_guidance(
            state=self.vehicle.state,
            threats_info=threat_status,
            nearest_sam_distance_m=nearest_sam_dist,
            dt=self.dt,
        )

        if self.guidance.bunker_destroyed:
            self.target_destroyed = True

        # 7. Check terminations & constraints
        terminated = False
        truncated = self.steps_count >= self.max_steps

        # SAM intercept check
        hit_by_sam = False
        for sam in self.iads.missiles:
            if sam.active:
                d_sam = math.sqrt(
                    (sam.x - self.vehicle.state.x) ** 2 +
                    (sam.y - self.vehicle.state.y) ** 2 +
                    (sam.z - self.vehicle.state.z) ** 2
                )
                if d_sam < 18.0 and not sam.spoofed_by_rgpo:
                    hit_by_sam = True
                    terminated = True
                    break

        # Ground impact or extreme structural crash
        if self.vehicle.state.z <= 0.0:
            terminated = True

        # Success termination: bunker destroyed and vehicle safe
        if self.target_destroyed and self.vehicle.state.x >= 70000.0:
            self.mission_success = True
            terminated = True

        # 8. Multi-Objective Reward Calculation
        reward = self._compute_reward(hit_by_sam, nearest_sam_dist)

        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_nearest_sam_distance(self) -> float | None:
        active_sams = [m for m in self.iads.missiles if m.active]
        if not active_sams:
            return None
        dists = [
            math.sqrt(
                (m.x - self.vehicle.state.x) ** 2 +
                (m.y - self.vehicle.state.y) ** 2 +
                (m.z - self.vehicle.state.z) ** 2
            )
            for m in active_sams
        ]
        return min(dists)

    def _compute_reward(self, hit_by_sam: bool, nearest_sam_dist: float | None) -> float:
        """Pareto multi-objective reward balancing survival, target strike, and physics safety."""
        reward = 0.0

        # Progress toward target (normalized)
        dist_to_bunker = math.sqrt(
            (self.guidance.target_pos[0] - self.vehicle.state.x) ** 2 +
            (self.guidance.target_pos[1] - self.vehicle.state.y) ** 2
        )
        reward += 0.05 * (self.vehicle.state.vx / 1500.0)

        # Target destroyed bonus
        if self.target_destroyed:
            reward += 150.0

        # Successful mission bonus
        if self.mission_success:
            reward += 300.0

        # Catastrophic failure penalties
        if hit_by_sam:
            reward -= 500.0
        if self.vehicle.state.z <= 100.0:
            reward -= 500.0

        # Structural G-load constraint penalty (zero violation rule: G <= 12.0)
        if self.vehicle.state.g_load > 12.0:
            excess = self.vehicle.state.g_load - 12.0
            reward -= 10.0 * (excess ** 2)

        # RGPO break-lock success reward
        if self.ew.state.radar_lock_broken:
            reward += 1.5

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        s = self.vehicle.state
        nearest_sam_dist = self._get_nearest_sam_distance() or 100000.0
        active_sams = [m for m in self.iads.missiles if m.active]
        sam_active = 1.0 if len(active_sams) > 0 else 0.0

        rgpo_phase_map = {"IDLE": 0.0, "CAPTURE": 0.33, "PULL_OFF": 0.66, "BREAK_LOCK": 1.0}
        rgpo_phase_val = rgpo_phase_map.get(self.ew.state.rgpo_phase, 0.0)

        dist_to_bunker = math.sqrt((self.guidance.target_pos[0] - s.x) ** 2 + (self.guidance.target_pos[1] - s.y) ** 2)
        bearing_to_bunker = math.atan2(self.guidance.target_pos[1] - s.y, self.guidance.target_pos[0] - s.x)

        obs = np.array([
            s.x / 100000.0, s.y / 10000.0, s.z / 50000.0,
            s.vx / 2000.0, s.vy / 500.0, s.vz / 500.0,
            s.roll, s.pitch, s.yaw,
            s.p, s.q, s.r,
            s.mass_fuel / max(1.0, self.vehicle.initial_fuel),
            s.g_load / 12.0,
            s.heat_flux_w_m2 / 1.5e6,
            1.0 if self.iads.radars[0].tracking_target else 0.0,
            1.0 if self.iads.radars[1].tracking_target else 0.0,
            1.0 if self.iads.radars[2].lock_on_active else 0.0,
            sam_active,
            nearest_sam_dist / 50000.0,
            0.0, 0.0,  # closing velocity & los rate placeholders
            self.ew.state.current_js_db / 30.0,
            rgpo_phase_val,
            1.0 if self.ew.state.radar_lock_broken else 0.0,
            dist_to_bunker / 100000.0,
            bearing_to_bunker,
            1.0 if self.target_destroyed else 0.0,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_info(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "mach": self.vehicle.state.mach,
            "altitude_m": self.vehicle.state.z,
            "g_load": self.vehicle.state.g_load,
            "fuel_kg": self.vehicle.state.mass_fuel,
            "ew_mode": self.ew.state.mode.value,
            "rgpo_phase": self.ew.state.rgpo_phase,
            "radar_lock_broken": self.ew.state.radar_lock_broken,
            "target_destroyed": self.target_destroyed,
            "mission_success": self.mission_success,
        }

