"""COG-EW offline evaluation entry point.

Evaluates an existing policy checkpoint and fails closed when loading,
benchmark execution, or metric validation fails.
"""

import argparse
import sys
from pathlib import Path

from sim.environment import HypersonicEWEnv
from rl.rl_agent import PPOAgent
from rl.evaluation import Evaluator, validate_metrics
from explain.explanation_generator import ExplanationGenerator

try:
    import torch
except ImportError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing COG-EW policy checkpoint."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Checkpoint path, absolute or relative to the project root.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")

    checkpoint = Path(args.model_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    checkpoint = checkpoint.resolve()

    print("=" * 74)
    print("COG-EW -- OFFLINE POLICY EVALUATION")
    print("=" * 74)
    print(f"Requested checkpoint: {checkpoint}")

    if not checkpoint.is_file():
        print("\n[ERROR] Checkpoint not found.")
        print("Evaluation has been cancelled; no benchmark was run.")
        print(
            "Provide an existing checkpoint using --model_path. "
            "File existence alone does not prove training occurred."
        )
        return 2

    if torch is None:
        print("\n[ERROR] PyTorch is required for policy evaluation.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        env = HypersonicEWEnv()
        state_dim = int(env.observation_space.shape[0])
        agent = PPOAgent(state_dim=state_dim, device=device)
        agent.load_checkpoint(str(checkpoint))
    except Exception as exc:
        print(f"\n[ERROR] Checkpoint or agent initialization failed: {exc}")
        print("Check the checkpoint format, dependencies, and policy architecture.")
        return 2

    print("[PASS] Checkpoint loaded.")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")

    try:
        evaluator = Evaluator(env=env, agent=agent)
        raw_metrics = evaluator.run_benchmark(num_episodes=args.episodes)
        metrics = validate_metrics(raw_metrics)
    except Exception as exc:
        print(f"\n[ERROR] Evaluation or metric validation failed: {exc}")
        return 1

    print("\nEvaluation results")
    print("-" * 74)
    print(f"Mission Success Rate: {metrics['mission_success_rate']:.1f}%")
    print(f"Survivability Index: {metrics['survivability_index']:.3f}")
    print("Mean Inference Latency: " f"{metrics['mean_inference_latency_ms']:.2f} ms")
    print("Max Inference Latency: " f"{metrics['max_inference_latency_ms']:.2f} ms")
    print(
        "Latency Constraint (<50 ms): "
        f"{'PASSED' if metrics['latency_constraint_met'] else 'FAILED'}"
    )

    # Explanation generation is supplementary and does not alter metrics.
    try:
        obs, _ = env.reset()
        action, _, _ = agent.select_action(obs, deterministic=True)
        explanation = ExplanationGenerator().generate_explanation(
            action,
            obs,
            [0.8, 0.1, 0.05, 0.05],
        )
        print("\nSample explanation")
        print("-" * 74)
        print(explanation.get("rationale", "No rationale was returned."))
    except Exception as exc:
        print(f"\n[WARNING] Explanation generation failed: {exc}")

    print("\nEvaluation completed.")
    print(
        "Note: successful checkpoint loading does not prove that "
        "PPO training occurred."
    )
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
