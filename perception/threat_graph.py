"""
Threat Knowledge Graph Representation (perception/threat_graph.py)
Graph representation & relational reasoning module for spatial and electromagnetic tactical environments.
"""

import networkx as nx
import numpy as np


class ThreatKnowledgeGraph:
    """
    Constructs a dynamic graph representing nodes (vehicle, search radar, track radar, target)
    and edges (illumination, missile link, jamming range, distance).
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def update_graph(self, vehicle_pos: np.ndarray, emitters: list, target_pos: np.ndarray):
        """Rebuild knowledge graph with current battlefield state."""
        self.graph.clear()

        # Add vehicle node
        self.graph.add_node("vehicle", type="hypersonic_asset", pos=vehicle_pos)
        
        # Add target node
        self.graph.add_node("target", type="mission_target", pos=target_pos)

        # Vehicle to target edge
        dist_to_target = np.linalg.norm(target_pos - vehicle_pos)
        self.graph.add_edge("vehicle", "target", relation="navigating_to", distance=dist_to_target)

        # Add emitter nodes & threat edges
        for e in emitters:
            e_id = e["id"]
            self.graph.add_node(e_id, type=e["type"], pos=e["position"])
            
            dist_to_vehicle = np.linalg.norm(vehicle_pos - e["position"])
            p_det = e.get("p_detection", 0.0)

            # Edge: Emitter illuminates Vehicle
            self.graph.add_edge(
                e_id, "vehicle",
                relation="illuminates",
                distance=dist_to_vehicle,
                p_detection=p_det,
                snr_db=e.get("snr_db", -100.0)
            )

    def get_highest_threat(self) -> str:
        """Find the emitter ID posing the highest detection threat to the vehicle."""
        max_p = -1.0
        top_emitter = None

        for u, v, data in self.graph.edges(data=True):
            if v == "vehicle" and data.get("relation") == "illuminates":
                if data.get("p_detection", 0.0) > max_p:
                    max_p = data.get("p_detection")
                    top_emitter = u

        return top_emitter
