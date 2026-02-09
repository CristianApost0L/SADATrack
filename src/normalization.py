"""
Skeleton normalization for view-invariant recognition.
"""
import numpy as np
from .constants import KP_L_HIP, KP_R_HIP, KP_L_SHOULDER, KP_R_SHOULDER


def normalize_skeleton(data):
    """
    Robust 2D Normalization for Scale-Invariant recognition.
    Crucial for handling players at different distances from camera.
    
    Operations:
    1. Centering: Hip Center → (0, 0)
    2. Scaling: Torso length → 1.0 (X, Y only - preserves confidence [0,1])
    
    Args:
        data: (C, T, V) numpy array where C=3 (X, Y, Confidence)
    
    Returns:
        Normalized skeleton data (3, T, V) as float32
    """
    # Ensure format (C, T, V)
    if data.shape[0] not in [2, 3, 4] and data.shape[2] in [2, 3, 4]:
         data = data.transpose(2, 0, 1) # Convert to C, T, V
         
    C, T, V = data.shape
    data_norm = data.copy()
    
    # 1. Centering (Subtract Hip Center) - X, Y only
    # Hip center = (Left Hip + Right Hip) / 2
    hip_center = (data[:2, :, KP_L_HIP] + data[:2, :, KP_R_HIP]) / 2.0  # (2, T)
    hip_center = np.expand_dims(hip_center, axis=-1) # (2, T, 1)
    
    data_norm[:2, :, :] = data[:2, :, :] - hip_center

    # 2. Scaling (Torso Size Invariance) - X, Y only
    # Torso length = Distance between Hip Center and Shoulder Center
    shoulder_center = (data_norm[:2, :, KP_L_SHOULDER] + data_norm[:2, :, KP_R_SHOULDER]) / 2.0
    # Hip center is 0,0 now (in XY)
    
    torso_len = np.linalg.norm(shoulder_center, axis=0) # (T,)
    mean_torso = np.mean(torso_len) + 1e-6 # Avoid div by zero
    
    # Safety Check: If torso is detected as too small (e.g. wrong detection or extreme angle), 
    # prevent massive scaling up of noise.
    if mean_torso < 0.1: # Arbitrary threshold in pixels, assuming reasonable resolution
         # Fallback: estimate scale from bounding box or just don't scale aggressively
         mean_torso = 1.0 

    # Scale ONLY X, Y coordinates - preserve confidence in [0, 1] range
    data_norm[:2, :, :] /= mean_torso
    
    # Robustness: Clip outliers from unstable pose detection
    # Normalized data typically in [-2, 2] range for body parts
    data_norm[:2, :, :] = np.clip(data_norm[:2, :, :], -5.0, 5.0)

    return data_norm.astype(np.float32)
