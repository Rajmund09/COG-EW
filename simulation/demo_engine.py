"""COG-EW synthetic mission simulation.

This module implements a closed, 2D demonstration environment:
- A virtual vehicle follows safe, predefined waypoints.
- Circular obstacles are displayed and checked for collisions.
- A beacon represents the virtual mission objective.
- Synthetic RF interference is displayed as monitoring telemetry only.
- No real-world RF transmission, weapon control, or engagement is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any


WORLD_WIDTH = 1000
WORLD_HEIGHT = 600
VEHICLE_SPEED = 85.0
ARRIVAL_RADIUS = 18.0
VEHICLE_RADIUS = 10.0
MAX_DT = 0.25

# The route is designed for this fixed, synthetic obstacle layout.
WAYPOINTS = [
    (220.0, 300.0),
    (280.0, 180.0),
    (430.0, 180.0),
    (450.0, 430.0),
    (620.0, 480.0),
    (790.0, 480.0),
    (850.0, 350.0),
    (900.0, 300.0),
]

OBSTACLES = [
    {"id": "obstacle_1", "x": 350.0, "y": 280.0, "radius": 60.0},
    {"id": "obstacle_2", "x": 540.0, "y": 300.0, "radius": 65.0},
    {"id": "obstacle_3", "x": 700.0, "y": 390.0, "radius": 50.0},
]

BEACON = {"x": 900.0, "y": 300.0, "radius": 22.0}


@dataclass
class VehicleState:
    x: float = 80.0
    y: float = 300.0
    heading_deg: float = 0.0
    speed: float = 0.0


class SimulationEngine:
    """Stateful engine for the COG-EW virtual mission demo."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Return the simulation to its initial state."""
        self.vehicle = VehicleState()
        self.status = "ready"
        self.elapsed_seconds = 0.0
        self.waypoint_index = 0
        self.interference_level = 0.0
        self.distance_travelled = 0.0
        self.events = ["Simulation initialized."]
        return self.get_state()

    def start(self) -> dict[str, Any]:
        """Start or resume the virtual mission."""
        if self.status in {"ready", "paused"}:
            self.status = "running"
            self._event("Virtual mission started.")
        elif self.status in {"completed", "collision"}:
            self._event("Reset the simulation before starting another mission.")
        return self.get_state()

    def pause(self) -> dict[str, Any]:
        """Pause movement without changing the mission state."""
        if self.status == "running":
            self.status = "paused"
            self._event("Simulation paused.")
        return self.get_state()

    def set_interference(self, level: float) -> dict[str, Any]:
        """Set a synthetic interference indicator for visualization.

        This parameter does not transmit RF energy and does not control
        vehicle navigation.
        """
        if isinstance(level, bool) or not isinstance(level, (int, float)):
            raise ValueError("Interference level must be numeric.")

        level = float(level)
        if not 0.0 <= level <= 1.0:
            raise ValueError("Interference level must be between 0 and 1.")

        previous = self.interference_level
        self.interference_level = level

        if level != previous:
            self._event(f"Synthetic RF interference display set to {level:.0%}.")

        return self.get_state()

    def step(
        self,
        dt: float = 0.1,
        interference_level: float | None = None,
    ) -> dict[str, Any]:
        """Advance the simulation by a bounded time interval."""
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise ValueError("dt must be numeric.")

        dt = float(dt)
        if not 0.0 < dt <= MAX_DT:
            raise ValueError(f"dt must be greater than 0 and at most {MAX_DT}.")

        if interference_level is not None:
            self.set_interference(interference_level)

        if self.status != "running":
            return self.get_state()

        remaining = VEHICLE_SPEED * dt

        # Move through the predefined safe route. Large steps can pass
        # multiple waypoints, but movement remains bounded by dt.
        while remaining > 0 and self.waypoint_index < len(WAYPOINTS):
            target_x, target_y = WAYPOINTS[self.waypoint_index]
            dx = target_x - self.vehicle.x
            dy = target_y - self.vehicle.y
            distance = hypot(dx, dy)

            if distance <= ARRIVAL_RADIUS:
                self.waypoint_index += 1
                continue

            self.vehicle.heading_deg = degrees(atan2(dy, dx))
            travel = min(remaining, distance)
            ratio = travel / distance

            old_x, old_y = self.vehicle.x, self.vehicle.y
            self.vehicle.x += dx * ratio
            self.vehicle.y += dy * ratio

            self.distance_travelled += hypot(
                self.vehicle.x - old_x,
                self.vehicle.y - old_y,
            )
            remaining -= travel

            if travel >= distance:
                self.waypoint_index += 1

        self.vehicle.speed = VEHICLE_SPEED
        self.elapsed_seconds += dt

        # Collision checks are for the virtual game environment only.
        for obstacle in OBSTACLES:
            separation = hypot(
                self.vehicle.x - obstacle["x"],
                self.vehicle.y - obstacle["y"],
            )
            if separation <= obstacle["radius"] + VEHICLE_RADIUS:
                self.status = "collision"
                self.vehicle.speed = 0.0
                self._event(f"Virtual collision with {obstacle['id']}.")
                break

        if self.status == "running":
            distance_to_beacon = hypot(
                self.vehicle.x - BEACON["x"],
                self.vehicle.y - BEACON["y"],
            )

            if distance_to_beacon <= BEACON["radius"]:
                self.status = "completed"
                self.vehicle.speed = 0.0
                self._event("Mission completed: beacon reached.")

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the dashboard."""
        distance_to_beacon = hypot(
            self.vehicle.x - BEACON["x"],
            self.vehicle.y - BEACON["y"],
        )

        if self.interference_level < 0.15:
            signal_quality = "GOOD"
        elif self.interference_level < 0.40:
            signal_quality = "DEGRADED"
        else:
            signal_quality = "POOR"

        progress = (
            100.0
            if self.status == "completed"
            else min(
                99.0,
                100.0 * self.waypoint_index / len(WAYPOINTS),
            )
        )

        return {
            "project": "COG-EW",
            "mode": "synthetic_simulation",
            "status": self.status,
            "world": {
                "width": WORLD_WIDTH,
                "height": WORLD_HEIGHT,
            },
            "vehicle": {
                "x": round(self.vehicle.x, 2),
                "y": round(self.vehicle.y, 2),
                "heading_deg": round(self.vehicle.heading_deg, 2),
                "speed": round(self.vehicle.speed, 2),
            },
            "mission": {
                "objective": "Reach the virtual beacon",
                "beacon": BEACON.copy(),
                "distance_remaining": round(distance_to_beacon, 2),
                "progress_percent": round(progress, 2),
                "waypoint_index": self.waypoint_index,
                "waypoint_count": len(WAYPOINTS),
            },
            "route": [list(point) for point in WAYPOINTS],
            "obstacles": [item.copy() for item in OBSTACLES],
            "telemetry": {
                "elapsed_seconds": round(self.elapsed_seconds, 2),
                "distance_travelled": round(self.distance_travelled, 2),
            },
            "rf_monitor": {
                "interference_level": self.interference_level,
                "interference_percent": round(self.interference_level * 100.0, 1),
                "signal_quality": signal_quality,
                "transmission_enabled": False,
            },
            "events": self.events[-8:],
        }

    def _event(self, message: str) -> None:
        self.events.append(message)
        self.events = self.events[-20:]
