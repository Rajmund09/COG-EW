"""
Multi-Static IADS Radar Network & SAM Interceptor Flight Dynamics
Simulates multi-band radars (VHF Search, S-Band Acquisition, X-Band Fire Control)
and supersonic Surface-to-Air Missiles with True Proportional Navigation Guidance.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RadarEmitter:
    """Ground-based air defense radar station."""
    name: str
    band: str                         # VHF, S-Band, X-Band
    role: str                         # SEARCH, ACQUISITION, FIRE_CONTROL
    x: float                          # Position x (m)
    y: float                          # Position y (m)
    z: float                          # Altitude z (m)
    range_max_m: float                # Maximum detection range (m)
    peak_power_kw: float
    antenna_gain_db: float
    beam_width_deg: float
    scan_azimuth_deg: float = 0.0
    scan_rate_deg_s: float = 60.0
    tracking_target: bool = False
    lock_on_active: bool = False
    lock_loss_timer: float = 0.0


@dataclass
class SAMInterceptor:
    """Supersonic Surface-to-Air Missile interceptor."""
    id: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    active: bool = True
    target_locked: bool = True
    spoofed_by_rgpo: bool = False
    motor_phase: str = "BOOST"        # BOOST -> SUSTAIN -> BURNOUT
    flight_time_s: float = 0.0
    max_lateral_g: float = 25.0       # Missile structural limit (25 G)
    nav_constant_n: float = 4.0       # Proportional navigation constant
    thrust_n: float = 45000.0
    mass_kg: float = 420.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)

    @property
    def mach(self) -> float:
        # Standard speed of sound at ~15km
        return self.speed / 300.0


class MultiStaticIADSNetwork:
    """
    Simulates integrated air defense radars and missile batteries.
    """

    def __init__(self):
        self.radars: list[RadarEmitter] = [
            RadarEmitter(
                name="Radar_EarlyWarning_VHF",
                band="VHF",
                role="SEARCH",
                x=18000.0,
                y=-6000.0,
                z=45.0,
                range_max_m=120000.0,
                peak_power_kw=250.0,
                antenna_gain_db=28.0,
                beam_width_deg=12.0,
                scan_rate_deg_s=36.0,
            ),
            RadarEmitter(
                name="Radar_Acquisition_SBand",
                band="S-Band",
                role="ACQUISITION",
                x=35000.0,
                y=4000.0,
                z=85.0,
                range_max_m=65000.0,
                peak_power_kw=180.0,
                antenna_gain_db=34.0,
                beam_width_deg=4.5,
                scan_rate_deg_s=60.0,
            ),
            RadarEmitter(
                name="Radar_FireControl_XBand",
                band="X-Band",
                role="FIRE_CONTROL",
                x=52000.0,
                y=-1200.0,
                z=120.0,
                range_max_m=42000.0,
                peak_power_kw=120.0,
                antenna_gain_db=42.0,
                beam_width_deg=1.4,
                scan_rate_deg_s=120.0,
            ),
        ]

        self.missiles: list[SAMInterceptor] = []
        self._sam_id_counter = 0

    def reset(self):
        for r in self.radars:
            r.tracking_target = False
            r.lock_on_active = False
            r.lock_loss_timer = 0.0
        self.missiles.clear()
        self._sam_id_counter = 0

    def launch_sam(self, target_pos: tuple[float, float, float]) -> SAMInterceptor:
        """Launch a supersonic SAM interceptor battery toward target coordinates."""
        self._sam_id_counter += 1
        fc = self.radars[2]  # Fire-control radar station
        sam = SAMInterceptor(
            id=f"SAM_SA21_{self._sam_id_counter}",
            x=fc.x + 300.0,
            y=fc.y + 150.0,
            z=fc.z + 10.0,
            vx=0.0,
            vy=0.0,
            vz=120.0,  # Vertical launch booster
            target_x=target_pos[0],
            target_y=target_pos[1],
            target_z=target_pos[2],
        )
        self.missiles.append(sam)
        return sam

    def step_threats(
        self,
        vehicle_pos: tuple[float, float, float],
        rgpo_phantom_pos: tuple[float, float, float] | None,
        rgpo_lock_broken: bool,
        noise_elevation_db: float,
        dt: float = 0.02,
    ) -> dict:
        """
        Update radar scan angles, target detections, and SAM missile guidance.
        """
        vx_v, vy_v, vz_v = vehicle_pos
        fc_radar = self.radars[2]

        # 1. Update radar scans and tracking
        for r in self.radars:
            r.scan_azimuth_deg = (r.scan_azimuth_deg + r.scan_rate_deg_s * dt) % 360.0
            dx = vx_v - r.x
            dy = vy_v - r.y
            dz = vz_v - r.z
            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            # Noise jamming reduces effective range: R_eff = R_max / 10^(noise_elevation / 40)
            eff_range = r.range_max_m / math.pow(10.0, noise_elevation_db / 40.0)

            if dist <= eff_range:
                r.tracking_target = True
                if r.role == "FIRE_CONTROL":
                    if not rgpo_lock_broken:
                        r.lock_on_active = True
                        r.lock_loss_timer = 0.0
                    else:
                        r.lock_loss_timer += dt
                        if r.lock_loss_timer > 1.2:
                            r.lock_on_active = False
            else:
                r.tracking_target = False
                r.lock_on_active = False

        # 2. Update active SAM missiles
        for sam in self.missiles:
            if not sam.active:
                continue

            sam.flight_time_s += dt

            # Motor thrust profile
            if sam.flight_time_s <= 4.5:
                sam.motor_phase = "BOOST"
                sam.thrust_n = 48000.0
                sam.mass_kg = 420.0 - 25.0 * sam.flight_time_s
            elif sam.flight_time_s <= 14.0:
                sam.motor_phase = "SUSTAIN"
                sam.thrust_n = 16000.0
                sam.mass_kg = 307.0 - 10.0 * (sam.flight_time_s - 4.5)
            else:
                sam.motor_phase = "BURNOUT"
                sam.thrust_n = 0.0
                sam.mass_kg = 212.0

            # Determine missile aiming point:
            # If RGPO is active and deceptive phantom echo pulls off, missile follows phantom!
            if rgpo_phantom_pos is not None and rgpo_lock_broken:
                aim_x, aim_y, aim_z = rgpo_phantom_pos
                sam.spoofed_by_rgpo = True
                sam.target_locked = False
            else:
                aim_x, aim_y, aim_z = vehicle_pos
                sam.spoofed_by_rgpo = False
                sam.target_locked = True

            # Line of Sight vector to target
            los_x = aim_x - sam.x
            los_y = aim_y - sam.y
            los_z = aim_z - sam.z
            range_to_target = math.sqrt(los_x ** 2 + los_y ** 2 + los_z ** 2)

            # True Proportional Navigation Guidance (PNG)
            # Closing velocity V_c = -(r_rel . v_rel) / R
            rel_vx = (aim_x - sam.target_x) / dt if sam.target_x != 0 else 0.0
            v_rel_x = rel_vx - sam.vx
            v_rel_y = -sam.vy
            v_rel_z = -sam.vz

            closing_velocity = -(los_x * v_rel_x + los_y * v_rel_y + los_z * v_rel_z) / max(1.0, range_to_target)

            # LOS rotation rate vector Omega = (r x v) / R^2
            omega_z = (los_x * v_rel_y - los_y * v_rel_x) / max(1.0, range_to_target ** 2)
            omega_y = (los_z * v_rel_x - los_x * v_rel_z) / max(1.0, range_to_target ** 2)

            # Acceleration command a_cmd = N * V_c * Omega
            a_cmd_x = sam.nav_constant_n * max(100.0, closing_velocity) * (-omega_y)
            a_cmd_y = sam.nav_constant_n * max(100.0, closing_velocity) * (omega_z)
            a_cmd_z = sam.nav_constant_n * max(100.0, closing_velocity) * (omega_y)

            # Apply missile maximum lateral acceleration limit (25 G)
            max_acc = sam.max_lateral_g * 9.80665
            cmd_mag = math.sqrt(a_cmd_x ** 2 + a_cmd_y ** 2 + a_cmd_z ** 2)
            if cmd_mag > max_acc:
                scale = max_acc / cmd_mag
                a_cmd_x *= scale
                a_cmd_y *= scale
                a_cmd_z *= scale

            # Aerodynamic drag on missile (approximate C_D ~ 0.35, S ~ 0.12 m^2)
            rho = 0.25  # Mid-stratospheric density
            q_sam = 0.5 * rho * (sam.speed ** 2)
            f_drag = q_sam * 0.12 * 0.35
            drag_ax = -f_drag * (sam.vx / max(1.0, sam.speed)) / sam.mass_kg
            drag_ay = -f_drag * (sam.vy / max(1.0, sam.speed)) / sam.mass_kg
            drag_az = -f_drag * (sam.vz / max(1.0, sam.speed)) / sam.mass_kg

            # Forward thrust component
            thrust_acc = sam.thrust_n / sam.mass_kg
            norm_v = max(1.0, sam.speed)

            # Integrate missile kinematics
            sam.vx += (a_cmd_x + drag_ax + thrust_acc * (sam.vx / norm_v)) * dt
            sam.vy += (a_cmd_y + drag_ay + thrust_acc * (sam.vy / norm_v)) * dt
            sam.vz += (a_cmd_z + drag_az + thrust_acc * (sam.vz / norm_v) - 9.80665) * dt

            sam.x += sam.vx * dt
            sam.y += sam.vy * dt
            sam.z += sam.vz * dt

            sam.target_x = aim_x
            sam.target_y = aim_y
            sam.target_z = aim_z

            # Terminate missile on ground hit or timeout
            if sam.z < 0.0 or sam.flight_time_s > 38.0:
                sam.active = False

        return {
            "radar_early_warning_tracking": self.radars[0].tracking_target,
            "radar_acquisition_tracking": self.radars[1].tracking_target,
            "radar_fire_control_locked": self.radars[2].lock_on_active,
            "active_missiles_count": len([m for m in self.missiles if m.active]),
        }

