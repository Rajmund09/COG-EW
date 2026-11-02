"""
Hypersonic Waverider Aerodynamics & Aerothermal Model
Computes lift, drag, pitching moment polars across subsonic, supersonic,
and hypersonic regimes, plus Sutton-Graves stagnation heat flux.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class AerodynamicForces:
    lift_n: float
    drag_n: float
    side_force_n: float
    c_l: float
    c_d: float
    c_y: float
    g_load: float
    dynamic_pressure_pa: float
    stagnation_heat_flux_w_m2: float
    mach: float


class HypersonicWaveriderAerodynamics:
    """
    Hypersonic waverider lifting body aerodynamic characteristics.
    Based on caret-wing waverider hypersonic wind-tunnel polars.
    """

    def __init__(
        self,
        reference_area_m2: float = 8.5,       # Wing reference planform area (m^2)
        nose_radius_m: float = 0.025,         # Leading edge bluntness radius (m)
        mass_dry_kg: float = 2400.0,          # Dry airframe mass (kg)
        sutton_graves_k: float = 1.7415e-4,   # Sutton-Graves constant for air in SI units
    ):
        self.s_ref = reference_area_m2
        self.r_nose = nose_radius_m
        self.mass_dry = mass_dry_kg
        self.k_sg = sutton_graves_k
        self.g0 = 9.80665

    def get_coefficients(self, alpha_rad: float, beta_rad: float, mach: float) -> tuple[float, float, float]:
        """
        Compute aerodynamic force coefficients: C_L, C_D, C_Y
        as functions of angle of attack (alpha), sideslip (beta), and Mach number.
        """
        alpha_deg = math.degrees(alpha_rad)
        beta_deg = math.degrees(beta_rad)

        # Baseline wave-drag variation with Mach number
        if mach < 0.8:
            cd0 = 0.015
            cl_alpha = 0.070  # per deg
        elif mach <= 1.2:
            # Transonic drag rise peak
            cd0 = 0.015 + 0.025 * ((mach - 0.8) / 0.4)
            cl_alpha = 0.065
        else:
            # Hypersonic decay proportional to 1 / sqrt(M^2 - 1)
            mach_factor = 1.0 / math.sqrt(max(0.1, mach ** 2 - 1.0))
            cd0 = 0.018 + 0.022 * min(1.0, mach_factor)
            cl_alpha = 0.045 * min(1.2, 2.0 * mach_factor + 0.4)

        # Non-linear hypersonic shock-expansion lift curve
        # C_L = C_L0 + C_Lalpha * sin(2*alpha) * cos(alpha)
        cl0 = 0.002
        cl = cl0 + cl_alpha * math.degrees(math.sin(2.0 * alpha_rad) * math.cos(alpha_rad))

        # Induced drag polar with cubic hypersonic cross-flow term
        # C_D = C_D0 + K * C_L^2 + C_Di * sin^3(|alpha|)
        k_induced = 0.28 / max(1.0, math.sqrt(mach))
        cd = cd0 + k_induced * (cl ** 2) + 1.20 * (abs(math.sin(alpha_rad)) ** 3)

        # Side-force from sideslip angle beta
        cy = -0.025 * beta_deg

        return cl, cd, cy

    def compute_forces(
        self,
        alpha_rad: float,
        beta_rad: float,
        airspeed_m_s: float,
        altitude_m: float,
        density_kg_m3: float,
        speed_of_sound_m_s: float,
        total_mass_kg: float,
    ) -> AerodynamicForces:
        """Calculate complete forces, dynamic pressure, G-load, and aerothermal heat flux."""
        mach = airspeed_m_s / max(1.0, speed_of_sound_m_s)
        q = 0.5 * density_kg_m3 * (airspeed_m_s ** 2)

        c_l, c_d, c_y = self.get_coefficients(alpha_rad, beta_rad, mach)

        lift = q * self.s_ref * c_l
        drag = q * self.s_ref * c_d
        side_force = q * self.s_ref * c_y

        # Instantaneous structural G-load (normal load factor n_z)
        g_load = abs(lift) / max(1.0, total_mass_kg * self.g0)

        # Sutton-Graves stagnation convective aerothermal heat flux (W/m^2)
        # q_dot = k_sg * sqrt(rho / R_nose) * V^3
        q_dot_stag = self.k_sg * math.sqrt(max(1e-8, density_kg_m3) / self.r_nose) * (airspeed_m_s ** 3)

        return AerodynamicForces(
            lift_n=lift,
            drag_n=drag,
            side_force_n=side_force,
            c_l=c_l,
            c_d=c_d,
            c_y=c_y,
            g_load=g_load,
            dynamic_pressure_pa=q,
            stagnation_heat_flux_w_m2=q_dot_stag,
            mach=mach,
        )

