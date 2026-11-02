"""
Continuous Hypersonic Dynamics & Explicit RK4 Numerical Integrator
Models continuous 3D/6-DoF trajectory kinematics, Earth gravity variation with altitude,
actuation moments, fuel mass depletion, and thermal accumulation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

from .atmosphere import USStandardAtmosphere1976
from .aerodynamics import HypersonicWaveriderAerodynamics, AerodynamicForces


@dataclass
class VehicleState:
    """Continuous state vector in R^14."""
    # Position: Downrange x (m), Crossrange y (m), Altitude z (m)
    x: float = 0.0
    y: float = 0.0
    z: float = 29500.0

    # Velocity components (m/s) in North-East-Down / Local Horizon
    vx: float = 1580.0    # ~Mach 5.2 at 29.5km
    vy: float = 0.0
    vz: float = 0.0

    # Euler attitude angles (radians)
    roll: float = 0.0     # Bank angle sigma
    pitch: float = 0.0    # Pitch angle theta
    yaw: float = 0.0      # Heading angle psi

    # Body angular rates (rad/s)
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0

    # Mass & Thermal
    mass_fuel: float = 650.0       # kg remaining
    thermal_load_kj: float = 0.0   # Accumulated heat (kJ)

    # Telemetry metrics (instantaneous)
    g_load: float = 1.0
    mach: float = 5.2
    dynamic_pressure_pa: float = 22000.0
    heat_flux_w_m2: float = 0.0
    airspeed: float = 1580.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw,
            self.p, self.q, self.r,
            self.mass_fuel, self.thermal_load_kj
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray, base: VehicleState | None = None) -> VehicleState:
        v = cls(
            x=float(arr[0]), y=float(arr[1]), z=float(arr[2]),
            vx=float(arr[3]), vy=float(arr[4]), vz=float(arr[5]),
            roll=float(arr[6]), pitch=float(arr[7]), yaw=float(arr[8]),
            p=float(arr[9]), q=float(arr[10]), r=float(arr[11]),
            mass_fuel=float(max(0.0, arr[12])),
            thermal_load_kj=float(arr[13]),
        )
        if base is not None:
            v.g_load = base.g_load
            v.mach = base.mach
            v.dynamic_pressure_pa = base.dynamic_pressure_pa
            v.heat_flux_w_m2 = base.heat_flux_w_m2
            v.airspeed = base.airspeed
        return v


@dataclass
class ControlInput:
    """Command inputs to vehicle actuators."""
    alpha_cmd_rad: float = 0.035        # Commanded Angle of Attack (~2.0 deg)
    bank_cmd_rad: float = 0.0           # Commanded bank angle sigma
    throttle: float = 0.0               # Scramjet / booster throttle [0, 1]
    rcs_yaw_torque: float = 0.0         # Reaction Control System yaw torque (N*m)


class ContinuousHypersonicVehicle:
    """
    Hypersonic Waverider vehicle with full aerodynamic, atmospheric,
    propulsion, and inertial physics.
    """

    R_EARTH = 6371000.0       # Earth mean radius (m)
    G0 = 9.80665              # Sea level gravitational acceleration (m/s^2)

    def __init__(
        self,
        mass_dry_kg: float = 2400.0,
        initial_fuel_kg: float = 650.0,
        max_thrust_n: float = 28000.0,
        isp_s: float = 1200.0,          # Air-breathing scramjet Isp (s)
    ):
        self.mass_dry = mass_dry_kg
        self.initial_fuel = initial_fuel_kg
        self.max_thrust = max_thrust_n
        self.isp = isp_s
        self.aero = HypersonicWaveriderAerodynamics(mass_dry_kg=mass_dry_kg)
        self.state = VehicleState(mass_fuel=initial_fuel_kg)

        # Vehicle Inertia Tensor (kg*m^2)
        self.Ixx = 2200.0
        self.Iyy = 14500.0
        self.Izz = 15800.0

    def get_gravity(self, altitude_m: float) -> float:
        """Inverse-square Earth gravitational acceleration."""
        h = max(0.0, altitude_m)
        return self.G0 * ((self.R_EARTH / (self.R_EARTH + h)) ** 2)

    def compute_derivatives(self, state_arr: np.ndarray, control: ControlInput) -> np.ndarray:
        """
        Compute continuous state derivatives f(t, s) for RK4 integration.
        Returns d(state)/dt as np.ndarray of shape (14,).
        """
        x, y, z = state_arr[0], state_arr[1], state_arr[2]
        vx, vy, vz = state_arr[3], state_arr[4], state_arr[5]
        roll, pitch, yaw = state_arr[6], state_arr[7], state_arr[8]
        p, q, r = state_arr[9], state_arr[10], state_arr[11]
        m_fuel = max(0.0, state_arr[12])

        total_mass = self.mass_dry + m_fuel
        altitude = max(10.0, z)
        airspeed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        airspeed = max(1.0, airspeed)

        # Atmospheric properties
        atm = USStandardAtmosphere1976.get_state(altitude)
        density = atm.density_kg_m3
        speed_of_sound = atm.speed_of_sound_m_s

        # Aerodynamic angles
        # Flight path angle gamma and ground track heading
        gamma = math.asin(np.clip(-vz / airspeed, -0.999, 0.999))
        track_heading = math.atan2(vy, vx)

        # Effective angle of attack: pitch minus flight path angle
        alpha = np.clip(control.alpha_cmd_rad, -np.radians(10.0), np.radians(22.0))
        bank = np.clip(control.bank_cmd_rad, -np.radians(75.0), np.radians(75.0))
        beta = 0.0

        # Aerodynamic forces
        forces = self.aero.compute_forces(
            alpha_rad=alpha,
            beta_rad=beta,
            airspeed_m_s=airspeed,
            altitude_m=altitude,
            density_kg_m3=density,
            speed_of_sound_m_s=speed_of_sound,
            total_mass_kg=total_mass,
        )

        # Propulsion thrust
        thrust = 0.0
        fuel_burn_rate = 0.0
        if m_fuel > 0.0 and control.throttle > 0.0:
            thrust = control.throttle * self.max_thrust
            fuel_burn_rate = thrust / (self.isp * self.G0)

        # Accelerations in wind / body coordinates resolved to local navigation frame
        # Lift is perpendicular to velocity vector, tilted by bank angle
        lift = forces.lift_n
        drag = forces.drag_n
        g_acc = self.get_gravity(altitude)

        # Dynamic equations of motion (point-mass 3D trajectory with bank-to-turn)
        # dV/dt = (T*cos(alpha) - D) / m - g*sin(gamma)
        # V * d(gamma)/dt = (L*cos(bank) + T*sin(alpha)) / m - g*cos(gamma)
        # V * cos(gamma) * d(heading)/dt = L * sin(bank) / m
        dv_dt = (thrust * math.cos(alpha) - drag) / total_mass - g_acc * math.sin(gamma)
        dgamma_dt = ((lift * math.cos(bank) + thrust * math.sin(alpha)) / total_mass - g_acc * math.cos(gamma)) / airspeed
        dheading_dt = (lift * math.sin(bank) / total_mass) / (airspeed * max(0.01, math.cos(gamma)))

        # Convert back to Cartesian velocity derivatives
        ax = dv_dt * math.cos(gamma) * math.cos(track_heading) - airspeed * math.sin(gamma) * math.cos(track_heading) * dgamma_dt - airspeed * math.cos(gamma) * math.sin(track_heading) * dheading_dt
        ay = dv_dt * math.cos(gamma) * math.sin(track_heading) - airspeed * math.sin(gamma) * math.sin(track_heading) * dgamma_dt + airspeed * math.cos(gamma) * math.cos(track_heading) * dheading_dt
        az = -(dv_dt * math.sin(gamma) + airspeed * math.cos(gamma) * dgamma_dt)  # -az because vz is down

        # Kinematics
        dx_dt = vx
        dy_dt = vy
        dz_dt = -vz  # vertical rate (positive up)

        # Attitude rate relaxation toward commanded angles
        tau_att = 0.15
        droll_dt = (bank - roll) / tau_att
        dpitch_dt = (gamma + alpha - pitch) / tau_att
        dyaw_dt = dheading_dt

        # Body rate damping
        dp_dt = -5.0 * p
        dq_dt = -5.0 * q
        dr_dt = -5.0 * r + control.rcs_yaw_torque / self.Izz

        # Consumables & thermal
        dm_fuel_dt = -fuel_burn_rate
        # Thermal load accumulation rate (kJ/s = kW)
        dQ_dt = forces.stagnation_heat_flux_w_m2 * (self.aero.s_ref * 0.12) / 1000.0

        return np.array([
            dx_dt, dy_dt, dz_dt,
            ax, ay, az,
            droll_dt, dpitch_dt, dyaw_dt,
            dp_dt, dq_dt, dr_dt,
            dm_fuel_dt, dQ_dt
        ], dtype=np.float64)


class RK4Integrator:
    """Explicit Runge-Kutta 4th Order numerical integrator."""

    @staticmethod
    def step(
        vehicle: ContinuousHypersonicVehicle,
        state: VehicleState,
        control: ControlInput,
        dt: float = 0.02,
    ) -> VehicleState:
        """
        Execute one explicit RK4 step:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        y0 = state.to_array()

        k1 = vehicle.compute_derivatives(y0, control)
        k2 = vehicle.compute_derivatives(y0 + 0.5 * dt * k1, control)
        k3 = vehicle.compute_derivatives(y0 + 0.5 * dt * k2, control)
        k4 = vehicle.compute_derivatives(y0 + dt * k3, control)

        y_next = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Update and compute instantaneous metrics
        next_state = VehicleState.from_array(y_next)

        # Compute derived metrics at updated state
        airspeed = math.sqrt(next_state.vx ** 2 + next_state.vy ** 2 + next_state.vz ** 2)
        atm = USStandardAtmosphere1976.get_state(next_state.z)
        mach = airspeed / max(1.0, atm.speed_of_sound_m_s)
        q = 0.5 * atm.density_kg_m3 * (airspeed ** 2)

        forces = vehicle.aero.compute_forces(
            alpha_rad=control.alpha_cmd_rad,
            beta_rad=0.0,
            airspeed_m_s=airspeed,
            altitude_m=next_state.z,
            density_kg_m3=atm.density_kg_m3,
            speed_of_sound_m_s=atm.speed_of_sound_m_s,
            total_mass_kg=vehicle.mass_dry + next_state.mass_fuel,
        )

        next_state.airspeed = airspeed
        next_state.mach = mach
        next_state.dynamic_pressure_pa = q
        next_state.g_load = forces.g_load
        next_state.heat_flux_w_m2 = forces.stagnation_heat_flux_w_m2

        return next_state

