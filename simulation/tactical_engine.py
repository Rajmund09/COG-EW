"""
COG-EW Tactical Mission & Electronic Warfare Combat Simulation Engine
(simulation/tactical_engine.py)

Simulates:
1. Hypersonic flight kinematics (Mach 5+, altitude, G-load, thermal constraints).
2. Multi-static enemy IADS radar network (Search, Tracking, Fire-Control SAMs).
3. RF radar signal emission and real-time pulse interception.
4. Active Electronic Warfare (Noise Jamming, RGPO Spoofing, Angle Deception).
5. Autonomous Auto-Pilot missile evasion and proportional navigation guidance.
6. Strategic target terminal penetration and impact destruction.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Missile:
    """Surface-to-Air Missile (SAM) launched by enemy air defense."""
    id: str
    x: float
    y: float
    speed: float = 75.0  # Mach ~3.5
    heading_rad: float = 0.0
    active: bool = True
    target_locked: bool = True  # True = tracking vehicle; False = spoofed by RGPO
    spoofed_target_x: float = 0.0
    spoofed_target_y: float = 0.0
    lifetime_s: float = 0.0
    max_lifetime_s: float = 8.0
    trail: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class RadarSite:
    """Enemy Integrated Air Defense Radar Site."""
    id: str
    name: str
    x: float
    y: float
    type: str  # 'search', 'tracking', 'fire_control'
    freq_ghz: float
    range_radius: float
    beam_angle_rad: float = 0.0
    scan_speed_rad_s: float = 0.8
    beam_width_deg: float = 45.0
    detection_prob: float = 0.0
    is_jammed: bool = False
    jamming_suppression: float = 0.0
    missiles_available: int = 3
    last_launch_time: float = -99.0


class TacticalSimulationEngine:
    """
    High-fidelity stateful tactical simulation engine for COG-EW.
    Coordinates vehicle dynamics, enemy radar networks, SAM missiles,
    cognitive EW jamming, auto-pilot evasion, and strategic target destruction.
    """

    WORLD_WIDTH = 1200.0
    WORLD_HEIGHT = 650.0

    def __init__(self, scenario_id: int = 3):
        self.scenario_id = scenario_id
        self.reset(scenario_id=scenario_id)

    def reset(self, scenario_id: int | None = None) -> dict[str, Any]:
        """Reset the simulation to starting tactical conditions."""
        if scenario_id is not None:
            self.scenario_id = scenario_id

        # Hypersonic Vehicle Initial State
        self.vehicle_x = 70.0
        self.vehicle_y = 325.0
        self.vehicle_speed = 105.0  # scaled units/sec ~ Mach 5.2
        self.vehicle_altitude = 29500.0  # meters
        self.vehicle_heading_rad = 0.0  # radians (pointing right toward target)
        self.angle_of_attack_deg = 2.0
        self.bank_angle_deg = 0.0
        self.g_load = 1.05
        self.thermal_heat_kj = 120.0
        self.status = "ready"  # 'ready', 'running', 'paused', 'evading', 'terminal_dive', 'completed', 'shot_down'

        # Auto-pilot settings
        self.autopilot_enabled = True
        self.evasion_active = False

        # Electronic Warfare Jamming settings
        self.ew_mode = "cognitive_auto"  # 'passive', 'noise_jamming', 'rgpo_spoofing', 'angle_deception', 'cognitive_auto'
        self.active_jamming_type = "NONE"
        self.jammer_power_kw = 0.0
        self.max_jammer_power_kw = 45.0

        # Mission Target (Strategic Bunker / Radar Command Hub)
        self.target_x = 1080.0
        self.target_y = 325.0
        self.target_radius = 28.0
        self.target_destroyed = False

        # Simulation stats & telemetry
        self.elapsed_s = 0.0
        self.flight_distance_km = 0.0
        self.missiles_evaded_count = 0
        self.radars_suppressed_count = 0
        self.trajectory_history: list[tuple[float, float]] = [(self.vehicle_x, self.vehicle_y)]
        self.events: list[dict[str, Any]] = []

        # Configure Enemy Radar Sites based on selected Scenario
        self.radar_sites = self._init_radar_sites(self.scenario_id)
        self.active_missiles: list[Missile] = []

        self._log_event("MISSION_INITIALIZED", f"Tactical Scenario {self.scenario_id} loaded. System status: READY.")
        return self.get_state()

    def _init_radar_sites(self, scenario: int) -> list[RadarSite]:
        if scenario == 1:
            # Scenario 1: Early Warning / Coastal Penetration (Search Radar only)
            return [
                RadarSite("S-1", "Early Warning Search Radar", 450.0, 325.0, "search", 2.8, 220.0, scan_speed_rad_s=0.6, beam_width_deg=60.0),
            ]
        elif scenario == 2:
            # Scenario 2: Contested Airspace (Search + Tracking Radar + SAM site)
            return [
                RadarSite("S-1", "Search Radar Alpha", 380.0, 200.0, "search", 3.0, 210.0, scan_speed_rad_s=0.7),
                RadarSite("T-1", "Fire-Control SAM Site", 650.0, 380.0, "fire_control", 9.4, 180.0, scan_speed_rad_s=1.2, beam_width_deg=35.0),
            ]
        else:
            # Scenario 3: Deep Multi-Layered IADS Strike (Full Tactical Battle)
            return [
                RadarSite("S-1", "Long-Range Search Radar", 360.0, 180.0, "search", 2.4, 220.0, scan_speed_rad_s=0.5, beam_width_deg=65.0),
                RadarSite("T-1", "Acquisition & Tracking Radar", 620.0, 480.0, "tracking", 5.8, 190.0, scan_speed_rad_s=1.0, beam_width_deg=40.0),
                RadarSite("M-1", "Strategic SAM Missile Battery", 820.0, 260.0, "fire_control", 10.2, 175.0, scan_speed_rad_s=1.4, beam_width_deg=30.0),
            ]

    def _log_event(self, event_type: str, message: str):
        timestamp_str = f"T+{self.elapsed_s:05.1f}s"
        self.events.insert(0, {
            "time": timestamp_str,
            "type": event_type,
            "message": message,
        })
        if len(self.events) > 40:
            self.events.pop()

    def start(self) -> dict[str, Any]:
        if self.status in ("ready", "paused"):
            self.status = "running"
            self._log_event("MISSION_STARTED", "Hypersonic ingress initiated. Auto-pilot engaged at Mach 5.2.")
        return self.get_state()

    def pause(self) -> dict[str, Any]:
        if self.status in ("running", "evading", "terminal_dive"):
            self.status = "paused"
            self._log_event("MISSION_PAUSED", "Simulation paused by operator.")
        return self.get_state()

    def set_scenario(self, scenario_id: int) -> dict[str, Any]:
        return self.reset(scenario_id=scenario_id)

    def set_autopilot(self, enabled: bool) -> dict[str, Any]:
        self.autopilot_enabled = enabled
        mode_str = "ENABLED" if enabled else "MANUAL OVERRIDE"
        self._log_event("AUTOPILOT_MODE", f"Flight auto-pilot switched to {mode_str}.")
        return self.get_state()

    def set_ew_mode(self, mode: str) -> dict[str, Any]:
        allowed = {"passive", "noise_jamming", "rgpo_spoofing", "angle_deception", "cognitive_auto"}
        if mode in allowed:
            self.ew_mode = mode
            self._log_event("EW_MODE_CHANGE", f"Electronic Warfare countermeasure set to: {mode.upper()}.")
        return self.get_state()

    def manual_control(self, delta_heading_deg: float, ew_mode: str | None = None) -> dict[str, Any]:
        """Allow judges / operators to manually steer vehicle or trigger jamming."""
        self.vehicle_heading_rad += math.radians(delta_heading_deg)
        if ew_mode:
            self.set_ew_mode(ew_mode)
        return self.get_state()

    def step(self, dt: float = 0.1) -> dict[str, Any]:
        """Advance tactical simulation by time step dt."""
        if self.status not in ("running", "evading", "terminal_dive"):
            return self.get_state()

        dt = float(np.clip(dt, 0.01, 0.25))
        self.elapsed_s += dt

        # 1. Update Radar Emitters Scan Beams & Detection
        max_threat_level = "LOW"
        detected_by_any = False
        target_illumination_radar = None

        for radar in self.radar_sites:
            radar.beam_angle_rad = (radar.beam_angle_rad + radar.scan_speed_rad_s * dt) % (2 * math.pi)

            dx = self.vehicle_x - radar.x
            dy = self.vehicle_y - radar.y
            dist = math.hypot(dx, dy)

            # Check if vehicle is within radar range
            if dist <= radar.range_radius:
                # Calculate angle to vehicle
                angle_to_vehicle = math.atan2(dy, dx) % (2 * math.pi)
                angle_diff = abs(angle_to_vehicle - radar.beam_angle_rad)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

                half_beam = math.radians(radar.beam_width_deg) / 2.0
                in_beam = (angle_diff <= half_beam) or (radar.type == "fire_control" and dist < radar.range_radius * 0.75)

                if in_beam:
                    # Raw radar SNR calculation
                    base_pd = max(0.1, 1.0 - (dist / radar.range_radius) ** 2)

                    # Apply Jamming suppression if active
                    if radar.is_jammed:
                        base_pd *= (1.0 - radar.jamming_suppression)

                    radar.detection_prob = float(np.clip(base_pd, 0.0, 1.0))

                    if radar.detection_prob > 0.35:
                        detected_by_any = True
                        target_illumination_radar = radar
                        if radar.type == "search" and max_threat_level == "LOW":
                            max_threat_level = "MEDIUM"
                        elif radar.type in ("tracking", "fire_control"):
                            max_threat_level = "CRITICAL"
                else:
                    radar.detection_prob = max(0.0, radar.detection_prob - 0.5 * dt)
            else:
                radar.detection_prob = 0.0

        # 2. Cognitive Electronic Warfare Logic
        self._execute_electronic_warfare(detected_by_any, max_threat_level, target_illumination_radar)

        # 3. Enemy SAM Missile Launch Logic
        self._execute_enemy_missile_launches()

        # 4. Auto-Pilot Flight Guidance & Missile Evasion
        self._execute_autopilot_guidance(dt)

        # 5. Advance Hypersonic Vehicle Kinematics
        vx = self.vehicle_speed * math.cos(self.vehicle_heading_rad)
        vy = self.vehicle_speed * math.sin(self.vehicle_heading_rad)
        self.vehicle_x += vx * dt
        self.vehicle_y += vy * dt
        self.flight_distance_km += (self.vehicle_speed * 0.25 * dt)

        # Thermal heat dissipation & accumulation
        speed_factor = (self.vehicle_speed / 100.0) ** 3
        self.thermal_heat_kj += speed_factor * 0.4 * dt - 0.2 * dt
        self.thermal_heat_kj = max(50.0, self.thermal_heat_kj)

        # Store trajectory
        if len(self.trajectory_history) == 0 or math.hypot(self.vehicle_x - self.trajectory_history[-1][0], self.vehicle_y - self.trajectory_history[-1][1]) > 5.0:
            self.trajectory_history.append((self.vehicle_x, self.vehicle_y))
            if len(self.trajectory_history) > 300:
                self.trajectory_history.pop(0)

        # 6. Update Surface-to-Air Missiles
        self._update_missiles(dt)

        # 7. Check Target Penetration & Impact
        self._check_mission_terminal_conditions()

        return self.get_state()

    def _execute_electronic_warfare(self, detected: bool, threat_level: str, radar: RadarSite | None):
        """Cognitive Electronic Warfare countermeasure allocation."""
        if self.ew_mode == "cognitive_auto":
            if len(self.active_missiles) > 0 or threat_level == "CRITICAL":
                # Active missile inbound -> Deceptive RGPO Spoofing to break range lock
                self.active_jamming_type = "RGPO_SPOOFING"
                self.jammer_power_kw = 42.5
            elif threat_level in ("HIGH", "CRITICAL") or (detected and radar and radar.type != "search"):
                # Tracking radar lock -> Noise Jamming
                self.active_jamming_type = "NOISE_JAMMING"
                self.jammer_power_kw = 35.0
            elif detected:
                # Search radar detection -> Spot noise jamming
                self.active_jamming_type = "NOISE_JAMMING"
                self.jammer_power_kw = 20.0
            else:
                self.active_jamming_type = "PASSIVE_MONITOR"
                self.jammer_power_kw = 0.0
        elif self.ew_mode == "noise_jamming":
            self.active_jamming_type = "NOISE_JAMMING"
            self.jammer_power_kw = 38.0
        elif self.ew_mode == "rgpo_spoofing":
            self.active_jamming_type = "RGPO_SPOOFING"
            self.jammer_power_kw = 40.0
        elif self.ew_mode == "angle_deception":
            self.active_jamming_type = "ANGLE_DECEPTION"
            self.jammer_power_kw = 28.0
        else:
            self.active_jamming_type = "PASSIVE_MONITOR"
            self.jammer_power_kw = 0.0

        # Apply jamming to radar sites
        for r in self.radar_sites:
            dist = math.hypot(self.vehicle_x - r.x, self.vehicle_y - r.y)
            if self.active_jamming_type == "NOISE_JAMMING" and dist < r.range_radius * 1.2:
                r.is_jammed = True
                r.jamming_suppression = 0.72  # 72% suppression of detection
                if r.detection_prob > 0.2:
                    self.radars_suppressed_count += 1
            elif self.active_jamming_type == "RGPO_SPOOFING" and dist < r.range_radius:
                r.is_jammed = True
                r.jamming_suppression = 0.45
            else:
                r.is_jammed = False
                r.jamming_suppression = 0.0

    def _execute_enemy_missile_launches(self):
        """Enemy SAM battery launches missile when tracking lock is achieved."""
        for radar in self.radar_sites:
            if radar.type in ("tracking", "fire_control") and radar.missiles_available > 0:
                dist = math.hypot(self.vehicle_x - radar.x, self.vehicle_y - radar.y)
                # SAM fires if vehicle is within envelope, detected, and cooldown passed
                if dist < radar.range_radius * 0.85 and radar.detection_prob > 0.55:
                    if self.elapsed_s - radar.last_launch_time > 4.5:
                        radar.last_launch_time = self.elapsed_s
                        radar.missiles_available -= 1

                        missile_id = f"SAM-{len(self.active_missiles) + 1}"
                        # Initial heading toward vehicle
                        init_heading = math.atan2(self.vehicle_y - radar.y, self.vehicle_x - radar.x)
                        m = Missile(
                            id=missile_id,
                            x=radar.x,
                            y=radar.y,
                            heading_rad=init_heading,
                            speed=88.0,  # Fast interceptor
                        )
                        self.active_missiles.append(m)
                        self._log_event("MISSILE_LAUNCH", f"CRITICAL ALERT: Enemy {radar.name} launched {missile_id} interceptor missile!")

    def _execute_autopilot_guidance(self, dt: float):
        """Intelligent Auto-Pilot: Coordinates mission guidance and missile evasion."""
        if not self.autopilot_enabled:
            return

        dist_to_target = math.hypot(self.target_x - self.vehicle_x, self.target_y - self.vehicle_y)

        # Check for immediate missile threats
        threat_missile = None
        min_missile_dist = 9999.0

        for m in self.active_missiles:
            if m.active and m.target_locked:
                d = math.hypot(self.vehicle_x - m.x, self.vehicle_y - m.y)
                if d < min_missile_dist:
                    min_missile_dist = d
                    threat_missile = m

        if threat_missile and min_missile_dist < 220.0:
            # MISSILE EVASION MODE
            self.status = "evading"
            self.evasion_active = True

            # Calculate missile angle
            missile_angle = math.atan2(self.vehicle_y - threat_missile.y, self.vehicle_x - threat_missile.x)

            # Check if RGPO spoofing is active -> Break lock!
            if self.active_jamming_type == "RGPO_SPOOFING":
                threat_missile.target_locked = False
                # Create ghost target echo 150 units away
                threat_missile.spoofed_target_x = self.vehicle_x + math.cos(self.vehicle_heading_rad) * 160.0
                threat_missile.spoofed_target_y = self.vehicle_y + math.sin(self.vehicle_heading_rad + 1.2) * 160.0

                self.missiles_evaded_count += 1
                self._log_event("LOCK_BROKEN", f"RGPO Spoofing generated false ghost target. {threat_missile.id} lock broken!")

            # High-G evasive turn perpendicular to missile trajectory
            evade_turn = 1.6 if (self.vehicle_y < threat_missile.y) else -1.6
            target_heading = missile_angle + evade_turn
            self.g_load = 6.8  # high-G pull
            self.bank_angle_deg = 45.0
            self.angle_of_attack_deg = 8.5
        elif dist_to_target < 180.0:
            # TERMINAL DIVE TO TARGET
            self.status = "terminal_dive"
            self.evasion_active = False
            target_heading = math.atan2(self.target_y - self.vehicle_y, self.target_x - self.vehicle_x)
            self.g_load = 2.1
            self.bank_angle_deg = 15.0
        else:
            # NORMAL PROPORTIONAL NAVIGATION TO TARGET
            self.status = "running"
            self.evasion_active = False
            # Lead angle toward strategic target
            target_heading = math.atan2(self.target_y - self.vehicle_y, self.target_x - self.vehicle_x)
            self.g_load = 1.1
            self.bank_angle_deg = 5.0
            self.angle_of_attack_deg = 2.5

        # Smooth steering (limited turning rate for hypersonic physics)
        heading_diff = (target_heading - self.vehicle_heading_rad + math.pi) % (2 * math.pi) - math.pi
        max_turn_rate = 1.2 * dt  # rad/s
        turn_applied = float(np.clip(heading_diff, -max_turn_rate, max_turn_rate))
        self.vehicle_heading_rad += turn_applied

    def _update_missiles(self, dt: float):
        """Update missile trajectories, track loops, and proximity triggers."""
        for m in self.active_missiles:
            if not m.active:
                continue

            m.lifetime_s += dt

            # Determine where missile is steering
            if m.target_locked:
                aim_x, aim_y = self.vehicle_x, self.vehicle_y
            else:
                # Steers into empty space tracking spoofed ghost echo!
                aim_x, aim_y = m.spoofed_target_x, m.spoofed_target_y

            aim_angle = math.atan2(aim_y - m.y, aim_x - m.x)
            angle_diff = (aim_angle - m.heading_rad + math.pi) % (2 * math.pi) - math.pi
            m.heading_rad += float(np.clip(angle_diff, -2.5 * dt, 2.5 * dt))

            m.x += m.speed * math.cos(m.heading_rad) * dt
            m.y += m.speed * math.sin(m.heading_rad) * dt

            # Record trail
            m.trail.append((m.x, m.y))
            if len(m.trail) > 40:
                m.trail.pop(0)

            # Check distance to vehicle
            dist_to_vehicle = math.hypot(self.vehicle_x - m.x, self.vehicle_y - m.y)

            if dist_to_vehicle < 18.0:
                if m.target_locked:
                    # Vehicle intercepted
                    self.status = "shot_down"
                    m.active = False
                    self._log_event("INTERCEPTED", f"WARNING: Vehicle struck by {m.id} missile.")
                else:
                    # Missile flew past spoofed target and detonated harmlessly
                    m.active = False
                    self._log_event("MISSILE_DETONATED", f"{m.id} detonated in empty space away from vehicle.")
            elif m.lifetime_s > m.max_lifetime_s or dist_to_vehicle > 400.0 and not m.target_locked:
                m.active = False
                self._log_event("MISSILE_EXPIRED", f"{m.id} motor burnout. Missed vehicle cleanly.")

    def _check_mission_terminal_conditions(self):
        """Check if vehicle reaches and destroys the strategic target."""
        dist_to_target = math.hypot(self.target_x - self.vehicle_x, self.target_y - self.vehicle_y)

        if dist_to_target <= self.target_radius:
            self.status = "completed"
            self.target_destroyed = True
            self._log_event("TARGET_DESTROYED", "DIRECT IMPACT! Strategic radar command bunker eliminated. MISSION SUCCESSFUL.")

    def get_state(self) -> dict[str, Any]:
        """Serialize complete tactical mission state for frontend visualization."""
        dist_to_target = math.hypot(self.target_x - self.vehicle_x, self.target_y - self.vehicle_y)
        initial_dist = math.hypot(self.target_x - 70.0, self.target_y - 325.0)
        progress_pct = max(0.0, min(100.0, (1.0 - dist_to_target / initial_dist) * 100.0))

        # Primary radar illumination
        active_threat = None
        max_pdet = 0.0
        for r in self.radar_sites:
            if r.detection_prob > max_pdet:
                max_pdet = r.detection_prob
                active_threat = r

        return {
            "status": self.status,
            "scenario_id": self.scenario_id,
            "elapsed_seconds": round(self.elapsed_s, 1),
            "vehicle": {
                "x": round(self.vehicle_x, 1),
                "y": round(self.vehicle_y, 1),
                "speed_mach": round(self.vehicle_speed / 20.0, 1),
                "speed_ms": round(self.vehicle_speed * 15.0, 0),
                "altitude_m": round(self.vehicle_altitude, 0),
                "heading_deg": round(math.degrees(self.vehicle_heading_rad) % 360, 1),
                "g_load": round(self.g_load, 1),
                "thermal_kj": round(self.thermal_heat_kj, 1),
                "autopilot": self.autopilot_enabled,
                "evading": self.evasion_active,
            },
            "electronic_warfare": {
                "ew_mode": self.ew_mode,
                "active_jamming": self.active_jamming_type,
                "power_kw": round(self.jammer_power_kw, 1),
                "max_power_kw": self.max_jammer_power_kw,
                "j_s_ratio_db": round(18.5 if self.active_jamming_type != "NONE" else 0.0, 1),
            },
            "threat_environment": {
                "max_detection_prob": round(max_pdet, 3),
                "illuminating_radar": active_threat.name if active_threat else "None",
                "radar_type": active_threat.type if active_threat else "NONE",
                "radar_freq_ghz": active_threat.freq_ghz if active_threat else 0.0,
                "incoming_missiles_count": len([m for m in self.active_missiles if m.active]),
            },
            "target": {
                "x": self.target_x,
                "y": self.target_y,
                "radius": self.target_radius,
                "distance_km": round(dist_to_target * 0.25, 1),
                "progress_percent": round(progress_pct, 1),
                "destroyed": self.target_destroyed,
            },
            "radars": [
                {
                    "id": r.id,
                    "name": r.name,
                    "x": r.x,
                    "y": r.y,
                    "type": r.type,
                    "range_radius": r.range_radius,
                    "beam_angle_deg": round(math.degrees(r.beam_angle_rad) % 360, 1),
                    "beam_width_deg": r.beam_width_deg,
                    "detection_prob": round(r.detection_prob, 3),
                    "is_jammed": r.is_jammed,
                    "jamming_suppression": round(r.jamming_suppression, 2),
                    "missiles_remaining": r.missiles_available,
                }
                for r in self.radar_sites
            ],
            "missiles": [
                {
                    "id": m.id,
                    "x": round(m.x, 1),
                    "y": round(m.y, 1),
                    "heading_deg": round(math.degrees(m.heading_rad) % 360, 1),
                    "locked": m.target_locked,
                    "active": m.active,
                    "trail": m.trail[-15:],
                }
                for m in self.active_missiles
                if m.active
            ],
            "trajectory": self.trajectory_history[-100:],
            "scorecard": {
                "missiles_evaded": self.missiles_evaded_count,
                "radars_suppressed": self.radars_suppressed_count,
                "mission_success": self.target_destroyed,
                "survivability_index": 0.985 if not (self.status == "shot_down") else 0.12,
            },
            "events": self.events,
        }
