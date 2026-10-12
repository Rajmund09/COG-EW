"""
Safety Projection Filter Module (safety/)
Enforces physical flight limits (g-force, thermal heating, minimum altitude).
"""

from safety.constraint_checker import SafetyConstraintFilter

__all__ = ["SafetyConstraintFilter"]
