"""
Main Evaluation Execution Script (evaluate.py)
Entry point for evaluating trained models and generating benchmark metrics.
"""

import argparse
import os
from sim.environment import HypersonicEWEnv
from rl.rl_agent import PPOAgent
from rl.evaluation import Evaluator
from explain.explanation_generator import ExplanationGenerator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cognitive EW-Guidance Agent")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of benchmark evaluation episodes")
    args = parser.parse_args()

    print("==========================================================================")
    print("     COGNITIVE MULTI-DOMAIN EW–GUIDANCE CO-DESIGNER (EVALUATION)         ")
    print("==========================================================================")

    env = HypersonicEWEnv()
    agent = PPOAgent(state_dim=16)

    if os.path.exists(args.model_path):
        agent.load_checkpoint(args.model_path)
        print(f"[+] Loaded trained model weights from {args.model_path}")
    else:
        print(f"[!] Warning: Model path {args.model_path} not found. Running evaluation with initialized policy weights.")

    evaluator = Evaluator(env=env, agent=agent)
    print(f"\n[+] Running {args.episodes} benchmark evaluation episodes...")
    metrics = evaluator.run_benchmark(num_episodes=args.episodes)

    print("\n---------------------- BENCHMARK EVALUATION RESULTS ----------------------")
    print(f"  Mission Success Rate       : {metrics['mission_success_rate']:.1f}%")
    print(f"  Survivability Index        : {metrics['survivability_index']:.3f}")
    print(f"  Mean Inference Latency     : {metrics['mean_inference_latency_ms']:.2f} ms")
    print(f"  Max Inference Latency      : {metrics['max_inference_latency_ms']:.2f} ms")
    print(f"  Real-time Constraint (<50ms): {'PASSED' if metrics['latency_constraint_met'] else 'FAILED'}")
    print("--------------------------------------------------------------------------")

    # Sample explanation generation
    obs, _ = env.reset()
    action, _, _ = agent.select_action(obs, deterministic=True)
    expl_gen = ExplanationGenerator()
    explanation = expl_gen.generate_explanation(action, obs, [0.8, 0.1, 0.05, 0.05])
    print(f"\n[+] Sample Tactical Rationale:\n    {explanation['rationale']}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
