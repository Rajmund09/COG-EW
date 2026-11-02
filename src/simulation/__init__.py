"""
COG-EW Aerospace-Grade Tactical Simulation Core
High-fidelity continuous flight dynamics, explicit RK4 integration,
closed-loop Proportional Navigation Guidance, and physics-based Cognitive EW.
"""

from .atmosphere import USStandardAtmosphere1976
from .aerodynamics import HypersonicWaveriderAerodynamics
from .dynamics import ContinuousHypersonicVehicle, RK4Integrator
from .actuators import ActuatorSuite
from .ew_physics import CognitiveEWPhysics
from .threats import MultiStaticIADSNetwork, SAMInterceptor
from .guidance import AutonomousTacticalGuidanceComputer
from .env import CogEWEnv

__all__ = [
    "USStandardAtmosphere1976",
    "HypersonicWaveriderAerodynamics",
    "ContinuousHypersonicVehicle",
    "RK4Integrator",
    "ActuatorSuite",
    "CognitiveEWPhysics",
    "MultiStaticIADSNetwork",
    "SAMInterceptor",
    "AutonomousTacticalGuidanceComputer",
    "CogEWEnv",
]

