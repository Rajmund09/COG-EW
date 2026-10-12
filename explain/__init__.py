"""
Explainability & Interpretability Module (explain/)
Attention visualizer, input feature saliency analyzer, and human-readable decision generator.
"""

from explain.attention_visualizer import AttentionVisualizer
from explain.saliency_analyzer import SaliencyAnalyzer
from explain.explanation_generator import ExplanationGenerator

__all__ = ["AttentionVisualizer", "SaliencyAnalyzer", "ExplanationGenerator"]
