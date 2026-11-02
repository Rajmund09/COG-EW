"""
US Standard Atmosphere 1976 Model
Covers altitudes from Sea Level (0 m) to 84 km with multi-layer lapse rates,
exact geopotential altitude conversion, dynamic pressure, and speed of sound.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AtmosphericState:
    altitude_m: float
    geopotential_alt_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
    dynamic_viscosity_pa_s: float


class USStandardAtmosphere1976:
    """1976 standard atmosphere model up to 84,000 meters."""

    R_EARTH = 6356766.0      # Nominal Earth radius for geopotential altitude (m)
    G0 = 9.80665              # Standard gravity at sea level (m/s^2)
    R_GAS = 287.05287         # Specific gas constant for dry air (J/(kg*K))
    GAMMA = 1.40              # Ratio of specific heats for air
    SUTHERLAND_C = 120.0      # Sutherland constant (K)
    BETA_S = 1.458e-6         # Sutherland coefficient (kg / (m*s*K^0.5))

    # Base heights (geopotential meters), lapse rates (K/m), base temperatures (K), base pressures (Pa)
    _LAYERS = [
        # (h_b, L_b, T_b, P_b)
        (0.0,     -0.0065,  288.15, 101325.0),
        (11000.0,  0.0,     216.65, 22632.06),
        (20000.0,  0.0010,  216.65, 5474.889),
        (32000.0,  0.0028,  228.65, 868.0187),
        (47000.0,  0.0,     270.65, 110.9063),
        (51000.0, -0.0028,  270.65, 66.93887),
        (71000.0, -0.0020,  214.65, 3.956420),
        (84852.0,  0.0,     186.87, 0.373383),
    ]

    @classmethod
    def get_state(cls, geometric_altitude_m: float) -> AtmosphericState:
        """Calculate complete atmospheric state at given geometric altitude."""
        h = max(0.0, float(geometric_altitude_m))

        # Geopotential altitude: H = (R_earth * h) / (R_earth + h)
        H = (cls.R_EARTH * h) / (cls.R_EARTH + h)

        # Clamp to model domain
        if H > 84852.0:
            H = 84852.0

        # Find layer
        idx = 0
        for i in range(len(cls._LAYERS) - 1):
            if H >= cls._LAYERS[i][0] and H < cls._LAYERS[i + 1][0]:
                idx = i
                break
        else:
            idx = len(cls._LAYERS) - 1

        h_b, L_b, T_b, P_b = cls._LAYERS[idx]
        dh = H - h_b

        if abs(L_b) < 1e-12:
            # Isothermal layer
            T = T_b
            P = P_b * math.exp(-cls.G0 * dh / (cls.R_GAS * T_b))
        else:
            # Non-isothermal layer
            T = T_b + L_b * dh
            P = P_b * math.pow(T_b / T, cls.G0 / (cls.R_GAS * L_b))

        # Ideal gas law: rho = P / (R * T)
        rho = P / (cls.R_GAS * T)

        # Speed of sound: a = sqrt(gamma * R * T)
        a = math.sqrt(cls.GAMMA * cls.R_GAS * T)

        # Dynamic viscosity (Sutherland's law)
        mu = (cls.BETA_S * math.pow(T, 1.5)) / (T + cls.SUTHERLAND_C)

        return AtmosphericState(
            altitude_m=h,
            geopotential_alt_m=H,
            temperature_k=T,
            pressure_pa=P,
            density_kg_m3=rho,
            speed_of_sound_m_s=a,
            dynamic_viscosity_pa_s=mu,
        )

    @classmethod
    def density(cls, altitude_m: float) -> float:
        return cls.get_state(altitude_m).density_kg_m3

    @classmethod
    def speed_of_sound(cls, altitude_m: float) -> float:
        return cls.get_state(altitude_m).speed_of_sound_m_s

    @classmethod
    def dynamic_pressure(cls, altitude_m: float, airspeed_m_s: float) -> float:
        rho = cls.density(altitude_m)
        return 0.5 * rho * (airspeed_m_s ** 2)

    @classmethod
    def mach(cls, altitude_m: float, airspeed_m_s: float) -> float:
        a = cls.speed_of_sound(altitude_m)
        return airspeed_m_s / max(1.0, a)

