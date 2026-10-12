"""
Dynamics Utilities & Frame Conversions (guidance/dynamics_utils.py)
Provides coordinate frame transformation math (NED, ECEF, Body-frame) and geometry helpers.
"""

import numpy as np


def ned_to_ecef(ned_pos: np.ndarray, ref_lat_lon_alt: np.ndarray) -> np.ndarray:
    """Convert North-East-Down (NED) position to Earth-Centered Earth-Fixed (ECEF)."""
    ref_lat = np.radians(ref_lat_lon_alt[0])
    ref_lon = np.radians(ref_lat_lon_alt[1])
    
    # Rotation matrix from NED to ECEF
    R = np.array([
        [-np.sin(ref_lat) * np.cos(ref_lon), -np.sin(ref_lon), -np.cos(ref_lat) * np.cos(ref_lon)],
        [-np.sin(ref_lat) * np.sin(ref_lon),  np.cos(ref_lon), -np.cos(ref_lat) * np.sin(ref_lon)],
        [ np.cos(ref_lat),                   0.0,             -np.sin(ref_lat)]
    ])
    
    ecef_offset = R @ ned_pos
    return ecef_offset


def ecef_to_ned(ecef_pos: np.ndarray, ref_lat_lon_alt: np.ndarray) -> np.ndarray:
    """Convert ECEF position to Local North-East-Down (NED)."""
    ref_lat = np.radians(ref_lat_lon_alt[0])
    ref_lon = np.radians(ref_lat_lon_alt[1])
    
    R = np.array([
        [-np.sin(ref_lat) * np.cos(ref_lon), -np.sin(ref_lat) * np.sin(ref_lon),  np.cos(ref_lat)],
        [-np.sin(ref_lon),                   np.cos(ref_lon),                    0.0],
        [-np.cos(ref_lat) * np.cos(ref_lon), -np.cos(ref_lat) * np.sin(ref_lon), -np.sin(ref_lat)]
    ])
    
    return R @ ecef_pos


def azimuth_elevation_distance(src_pos: np.ndarray, dst_pos: np.ndarray) -> tuple:
    """Compute Azimuth (rad), Elevation (rad), and Distance (m) between two points."""
    rel = dst_pos - src_pos
    distance = np.linalg.norm(rel)
    azimuth = np.arctan2(rel[1], rel[0])
    
    # Assume 2D/3D z coordinate for elevation
    z = rel[2] if len(rel) > 2 else 0.0
    elevation = np.arcsin(z / max(distance, 1e-6))
    
    return azimuth, elevation, distance
