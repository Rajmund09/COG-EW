"""
Hypersonic Vehicle Dynamics Model (sim/vehicle_dynamics.py)
Provides 2D point-mass flight kinematics, aerodynamics, atmospheric models, and constraint checking.
"""

import numpy as np


class HypersonicVehicle:
    """
    2D point-mass flight dynamics for hypersonic vehicle.
    State vector: [x, y, v, flight_path_angle (gamma), mass, thermal_state]
    """

    def __init__(self, mass: float = 1000.0, area: float = 0.5, mach_init: float = 6.0):
        self.mass = mass
        self.reference_area = area
        self.g0 = 9.80665  # m/s^2
        self.r_earth = 6371000.0  # m
        
        # Initial conditions
        self.init_mach = mach_init
        self.reset()

    def reset(self, initial_pos: np.ndarray = None, altitude: float = 30000.0):
        """Reset vehicle state to initial parameters."""
        if initial_pos is None:
            initial_pos = np.array([0.0, altitude], dtype=np.float64)
        
        # Velocity from Mach number assuming speed of sound at 30km (~300 m/s)
        self.speed_of_sound = 301.0
        v_init = self.init_mach * self.speed_of_sound

        self.state = {
            "position": initial_pos,             # [x (m), y (altitude m)]
            "velocity": v_init,                  # scalar speed (m/s)
            "heading": 0.0,                      # radians gamma
            "angle_of_attack": 0.0,              # radians alpha
            "bank_angle": 0.0,                   # radians sigma
            "mass": self.mass,
            "thermal_heat": 0.0,                 # accumulated thermal load (kJ)
            "g_load": 1.0,                       # instantaneous g-load
        }
        return self.get_kinematic_state()

    def get_atmospheric_density(self, altitude: float) -> float:
        """Exponential atmospheric density model (kg/m^3)."""
        rho0 = 1.225
        scale_height = 7200.0
        return rho0 * np.exp(-max(0.0, altitude) / scale_height)

    def step(self, action_guidance: np.ndarray, dt: float = 0.1) -> dict:
        """
        Update vehicle dynamics for time step dt.
        action_guidance: [delta_angle_of_attack, bank_angle]
        """
        alpha = np.clip(action_guidance[0], np.radians(-15), np.radians(20))
        sigma = np.clip(action_guidance[1], np.radians(-60), np.radians(60))

        x, y = self.state["position"]
        v = self.state["velocity"]
        gamma = self.state["heading"]

        rho = self.get_atmospheric_density(y)
        q = 0.5 * rho * (v ** 2)  # Dynamic pressure

        # Aerodynamic coefficients (simplified hypersonic lift/drag curves)
        cL = 1.5 * np.sin(2 * alpha)
        cD = 0.05 + 1.2 * (np.sin(alpha) ** 2)

        lift = q * self.reference_area * cL
        drag = q * self.reference_area * cD

        # Equations of motion (2D point-mass with gravity)
        g = self.g0 * (self.r_earth / (self.r_earth + y)) ** 2
        
        dv_dt = -drag / self.mass - g * np.sin(gamma)
        dgamma_dt = (lift * np.cos(sigma) / (self.mass * max(v, 1.0))) - (g * np.cos(gamma) / max(v, 1.0))

        # Kinematic updates
        v_next = max(100.0, v + dv_dt * dt)
        gamma_next = gamma + dgamma_dt * dt
        x_next = x + v * np.cos(gamma) * dt
        y_next = max(0.0, y + v * np.sin(gamma) * dt)

        # Thermal heating model (Sutton-Grave formula approximation)
        heat_flux = 1.83e-4 * np.sqrt(rho) * (v ** 3)  # kW/m^2
        self.state["thermal_heat"] += heat_flux * dt

        # G-load calculation
        L_acc = lift / (self.mass * self.g0)
        self.state["g_load"] = np.sqrt(1.0 + L_acc ** 2)

        self.state["position"] = np.array([x_next, y_next], dtype=np.float64)
        self.state["velocity"] = v_next
        self.state["heading"] = gamma_next
        self.state["angle_of_attack"] = alpha
        self.state["bank_angle"] = sigma

        return self.get_kinematic_state()

    def get_kinematic_state(self) -> np.ndarray:
        """Return standardized kinematic state vector."""
        return np.array([
            self.state["position"][0],
            self.state["position"][1],
            self.state["velocity"],
            self.state["heading"],
            self.state["g_load"],
            self.state["thermal_heat"]
        ], dtype=np.float32)
