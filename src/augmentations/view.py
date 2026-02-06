"""
View-based augmentations (3D rotations and camera simulations).
"""
import numpy as np


def augment_3d_view_rotation(data, angle_range=30):
    """
    Simulates camera moving around the player (Y-axis rotation).
    Requires 3D data (C >= 3).
    
    Args:
        data: (C, T, V) skeleton data
        angle_range: rotation range in degrees
    
    Returns:
        3D rotated skeleton data
    """
    C, T, V = data.shape
    if C < 3: 
        return data # Cannot do 3D rotation without Z
    
    angle = np.radians(np.random.uniform(-angle_range, angle_range))
    c, s = np.cos(angle), np.sin(angle)
    
    # Rotation around Y-axis
    # x' = x*cos + z*sin
    # z' = -x*sin + z*cos
    
    x = data[0].copy()
    z = data[2].copy()
    
    data[0] = x * c + z * s
    data[2] = -x * s + z * c
    
    return data


def simulate_back_view(data):
    """
    Simulates a strict Back-View (180 degree rotation).
    Useful to mitigate bias when training only on frontal data.
    
    Transformation: 
    - 3D (with Z): (x, y, z) -> (-x, y, -z)
    - 2D (no Z): (x, y) -> (-x, y) [Approximation! Mirrors the player]
    
    Args:
        data: (C, T, V) skeleton data
    
    Returns:
        Back-view transformed skeleton data
    """
    C, T, V = data.shape
    
    # 180 degree rotation around Y-axis
    if C >= 3:
        # Full 3D rotation: x' = -x, z' = -z
        data[0] = -data[0]
        data[2] = -data[2]
    else:
        # 2D Approximation: Mirroring X
        # Note: This creates a "Back View" projection, BUT it mirrors the chirality (Righty -> Lefty).
        # Since we likely apply random horizontal flip afterwards anyway, this is acceptable
        # for data augmentation purposes to simulate "swinging away from camera".
        data[0] = -data[0]
    
    return data
