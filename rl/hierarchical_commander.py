"""
Hierarchical Cognitive Commander (rl/hierarchical_commander.py)
Strategic Commander AI coordinating Guidance, EW, and Resource specialists.
"""

import numpy as np


class HierarchicalCognitiveCommander:
    """
    Top-level Strategic Commander that selects sub-domain goals and resolves domain conflicts
    between guidance trajectory optimization and electronic warfare power consumption.
    """

    def __init__(self, agent):
        self.agent = agent

    def coordinate_tactical_action(self, situation_vector: np.ndarray) -> dict:
        """
        Coordinates specialists:
        1. Strategic Goal Selection (Penetrate / Evade / Attack)
        2. Specialist Action Proposals (Guidance vs EW)
        3. Conflict Resolution
        """
        action, log_prob, value = self.agent.select_action(situation_vector)

        # Conflict resolution logic: If high threat detected, prioritize EW over trajectory efficiency
        max_p_det = situation_vector[8] if len(situation_vector) > 8 else 0.0
        
        guidance_cmd = action[0:2]
        ew_cmd = action[2:4]

        if max_p_det > 0.8:
            # Boost EW power action to max for emergency defense
            ew_cmd[1] = 1.0

        return {
            "final_action": np.concatenate([guidance_cmd, ew_cmd]),
            "commander_value": float(value),
            "threat_override_active": max_p_det > 0.8
        }
