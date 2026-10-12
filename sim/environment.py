"""
Gymnasium-Compliant POMDP Environment (sim/environment.py)
Joint Hypersonic Guidance & Electronic Warfare Simulation Environment.
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        class BaseEnv:
            pass
        class spaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=None):
                    self.low = low
                    self.high = high
                    self.shape = shape or (low.shape if hasattr(low, 'shape') else None)
                    self.dtype = dtype
        gym = type('gym', (), {'Env': BaseEnv})
import numpy as np

from sim.vehicle_dynamics import HypersonicVehicle
from sim.emitter_models import EmitterNetwork
from sim.ew_effects import EWEffectsModel


class HypersonicEWEnv(gym.Env):
    """
    OpenAI Gym environment for joint Guidance and EW control.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: dict = None):
        super().__init__()
        self.config = env_config or {}

        # Initialize sub-modules
        self.vehicle = HypersonicVehicle()
        self.emitters = EmitterNetwork()
        self.ew_model = EWEffectsModel()

        self.target_pos = np.array([300000.0, 0.0], dtype=np.float64)
        self.max_steps = 1000
        self.current_step = 0

        # Action Space: [guidance_alpha_delta, guidance_sigma, ew_mode, ew_power_ratio]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 3.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space: [vehicle_state (6), rel_target_dist, rel_target_angle, radar_threat_summary (8)] -> total 16
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if hasattr(super(), 'reset') and callable(getattr(super(), 'reset', None)):
            try:
                super().reset(seed=seed)
            except Exception:
                pass
        self.current_step = 0
        kinematic_state = self.vehicle.reset()
        self.emitters.update(0.0)
        obs = self._build_observation(kinematic_state)
        return obs, {}

    def step(self, action: np.ndarray):
        self.current_step += 1

        # Unpack actions
        guidance_act = action[0:2]
        ew_act = action[2:4]

        # 1. Update vehicle dynamics
        kinematic_state = self.vehicle.step(guidance_act, dt=0.1)
        vehicle_pos = self.vehicle.state["position"]

        # 2. Compute EW countermeasures
        threat_eval_pre = self.emitters.evaluate_threats(vehicle_pos)
        ew_impacts = self.ew_model.compute_countermeasures(ew_act, vehicle_pos, threat_eval_pre)

        # 3. Update emitters with jamming effects
        self.emitters.update(0.1)
        threat_eval_post = self.emitters.evaluate_threats(
            vehicle_pos, jamming_map=ew_impacts["jamming_noise_db"]
        )

        # 4. Calculate reward & check terminal conditions
        reward, terminated, truncated, info = self._compute_reward_and_terminal(
            vehicle_pos, threat_eval_post, ew_impacts
        )

        obs = self._build_observation(kinematic_state, threat_eval_post)
        return obs, reward, terminated, truncated, info

    def _build_observation(self, kinematic_state: np.ndarray, threat_eval: list = None) -> np.ndarray:
        vehicle_pos = self.vehicle.state["position"]
        target_vec = self.target_pos - vehicle_pos
        rel_dist = np.linalg.norm(target_vec)
        rel_angle = np.arctan2(target_vec[1], target_vec[0])

        # Aggregate threat features
        max_p_det = 0.0
        max_snr = -100.0
        if threat_eval:
            max_p_det = max([t["p_detection"] for t in threat_eval])
            max_snr = max([t["snr_db"] for t in threat_eval])

        threat_summary = np.array([
            max_p_det, max_snr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

        return np.concatenate([
            kinematic_state.astype(np.float32),
            np.array([rel_dist, rel_angle], dtype=np.float32),
            threat_summary
        ], dtype=np.float32)

    def _compute_reward_and_terminal(self, pos: np.ndarray, threats: list, ew_impacts: dict):
        dist_to_target = np.linalg.norm(pos - self.target_pos)
        
        # 1. Distance progress reward
        r_dist = -dist_to_target / 10000.0

        # 2. Survivability penalty based on radar detection probability
        max_p_det = max([t["p_detection"] for t in threats]) if threats else 0.0
        r_survive = -10.0 * max_p_det

        # 3. Energy penalty for active jamming
        r_energy = -0.1 * ew_impacts.get("power_consumed_kw", 0.0)

        reward = r_dist + r_survive + r_energy

        terminated = False
        truncated = False
        info = {"max_p_detection": max_p_det, "dist_to_target": dist_to_target}

        if dist_to_target < 100.0:
            reward += 1000.0
            terminated = True
            info["status"] = "target_reached"
        elif self.vehicle.state["position"][1] < 0.0 or self.vehicle.state["g_load"] > 10.0:
            reward -= 500.0
            terminated = True
            info["status"] = "crash_or_overload"
        elif self.current_step >= self.max_steps:
            truncated = True
            info["status"] = "max_steps_exceeded"

        return float(reward), terminated, truncated, info
