"""
Comprehensive Unit Tests for COG-EW RK4 Aerospace Simulation Core
"""

import math
import numpy as np
import pytest

from src.simulation.atmosphere import USStandardAtmosphere1976
from src.simulation.aerodynamics import HypersonicWaveriderAerodynamics
from src.simulation.dynamics import ContinuousHypersonicVehicle, VehicleState, ControlInput, RK4Integrator
from src.simulation.ew_physics import CognitiveEWPhysics, EWJammingMode
from src.simulation.threats import MultiStaticIADSNetwork
from src.simulation.guidance import AutonomousTacticalGuidanceComputer, MissionPhase
from src.simulation.env import CogEWEnv


def test_atmosphere_sea_level_and_stratosphere():
    """Verify 1976 standard atmosphere density, pressure, and sound speed."""
    sl = USStandardAtmosphere1976.get_state(0.0)
    assert abs(sl.pressure_pa - 101325.0) < 1.0
    assert abs(sl.temperature_k - 288.15) < 0.1
    assert abs(sl.density_kg_m3 - 1.225) < 0.01
    assert abs(sl.speed_of_sound_m_s - 340.29) < 0.5

    # Stratosphere at 30,000m
    strato = USStandardAtmosphere1976.get_state(30000.0)
    assert strato.temperature_k < 235.0
    assert strato.density_kg_m3 < 0.02
    assert strato.speed_of_sound_m_s < 310.0


def test_aerodynamics_hypersonic_lift_and_heating():
    """Verify wave-drag polar and Sutton-Graves stagnation heat flux."""
    aero = HypersonicWaveriderAerodynamics()
    forces = aero.compute_forces(
        alpha_rad=math.radians(4.0),
        beta_rad=0.0,
        airspeed_m_s=1580.0,
        altitude_m=29500.0,
        density_kg_m3=0.018,
        speed_of_sound_m_s=302.0,
        total_mass_kg=3050.0,
    )
    assert forces.mach > 5.0
    assert forces.lift_n > 0.0
    assert forces.drag_n > 0.0
    assert forces.c_l / forces.c_d > 2.5  # Hypersonic Waverider L/D efficiency
    assert forces.stagnation_heat_flux_w_m2 > 10000.0


def test_rk4_numerical_integration_step():
    """Verify explicit RK4 step smoothly updates position and velocity."""
    veh = ContinuousHypersonicVehicle()
    init_state = VehicleState(x=0.0, y=0.0, z=29500.0, vx=1580.0, vy=0.0, vz=0.0)
    control = ControlInput(alpha_cmd_rad=math.radians(2.0), bank_cmd_rad=0.0, throttle=0.35)

    next_state = RK4Integrator.step(veh, init_state, control, dt=0.02)
    # Downrange position advance: x = vx * dt = 1580 * 0.02 = 31.6m
    assert next_state.x > 30.0
    assert abs(next_state.z - 29500.0) < 5.0
    assert next_state.mach > 5.0
    assert next_state.g_load <= 12.0  # Zero G violation


def test_ew_rgpo_pull_off_and_break_lock():
    """Verify RGPO state machine transitions to break lock."""
    ew = CognitiveEWPhysics(rgpo_gate_width_m=100.0, rgpo_pull_rate_m_s2=200.0)

    # 1. Passive ESM
    s1 = ew.step(EWJammingMode.PASSIVE_ESM, distance_to_radar_m=25000.0, radar_locked=True, dt=0.02)
    assert s1.rgpo_phase == "IDLE"
    assert not s1.radar_lock_broken

    # 2. Step through RGPO Deception
    ew.step(EWJammingMode.RGPO_DECEPTION, distance_to_radar_m=25000.0, radar_locked=True, dt=0.5)
    assert ew.state.rgpo_phase in ("CAPTURE", "PULL_OFF")

    # Step past pull-off time to trigger gate drop
    for _ in range(150):
        ew.step(EWJammingMode.RGPO_DECEPTION, distance_to_radar_m=25000.0, radar_locked=True, dt=0.02)

    assert ew.state.radar_lock_broken
    assert ew.state.rgpo_apparent_range_offset_m > 100.0


def test_gymnasium_environment_api():
    """Verify Gymnasium environment reset and step compliance."""
    env = CogEWEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (28,)
    assert isinstance(info, dict)
    assert info["mach"] > 4.5
    assert info["altitude_m"] > 28000.0

    action = np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32)  # RGPO mode command
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert obs2.shape == (28,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not np.isnan(obs2).any()
    assert info2["g_load"] <= 12.0
    env.close()

