"""
Human-Readable Explanation Generator (explain/explanation_generator.py)
Generates natural language rationale and confidence scores for tactical EW and Guidance decisions.
"""

import numpy as np


class ExplanationGenerator:
    """
    Translates neural network policy outputs and attention maps into human-readable operator explanations.
    """

    def __init__(self):
        self.mode_names = ["Passive Listening", "Active Noise Jamming", "Range Gate Pull-Off Spoofing", "Angle Deception"]

    def generate_explanation(self, action: np.ndarray, situation_vector: np.ndarray, attention_weights: np.ndarray) -> dict:
        mode_idx = int(action[2]) if len(action) > 2 else 0
        power_ratio = float(action[3]) if len(action) > 3 else 0.0
        mode_name = self.mode_names[min(mode_idx, 3)]

        top_emitter_idx = int(np.argmax(attention_weights)) if len(attention_weights) > 0 else 0
        confidence = float(np.max(attention_weights)) if len(attention_weights) > 0 else 0.85

        rationale = (
            f"Tactical Action: Executing {mode_name} at {power_ratio*100:.1f}% power level. "
            f"Primary Threat Focus: Radar Emitter #{top_emitter_idx + 1} due to high illumination risk. "
            f"Guidance Maneuver: Angle-of-Attack delta {action[0]:.3f} rad, Bank angle {action[1]:.3f} rad."
        )

        return {
            "rationale": rationale,
            "confidence_score": round(confidence, 3),
            "chosen_ew_mode": mode_name,
            "target_emitter_id": f"emitter_{top_emitter_idx + 1}"
        }
