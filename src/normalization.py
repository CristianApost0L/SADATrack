"""
Skeleton normalization for view-invariant recognition.
"""
import numpy as np
from .constants import KP_L_HIP, KP_R_HIP, KP_L_SHOULDER, KP_R_SHOULDER


def normalize_skeleton(data):
    """
    Robust 3D/2D Normalization for View-Invariant recognition.
    Crucial for ATP/Broadcast footage.
    
    Operations:
    1. Centering: Hip Center -> (0,0,0)
    2. Rotation Alignment (Canonic View): Rotates skeleton so hips align with X-axis (DISABLED for robustness)
    3. Scaling: Torso length -> 1.0
    
    Args:
        data: (C, T, V) or (T, V, C) numpy array. Assumed (C, T, V) based on project.
    
    Returns:
        Normalized skeleton data (C, T, V) as float32
    """
    # Ensure format (C, T, V)
    if data.shape[0] not in [2, 3, 4] and data.shape[2] in [2, 3, 4]:
         data = data.transpose(2, 0, 1) # Convert to C, T, V
         
    C, T, V = data.shape
    data_norm = data.copy()
    
    # 1. Centering (Subtract Hip Center)
    # Hip center = (Left Hip + Right Hip) / 2
    hip_center = (data[:2, :, KP_L_HIP] + data[:2, :, KP_R_HIP]) / 2.0  # (2, T)
    hip_center = np.expand_dims(hip_center, axis=-1) # (2, T, 1)
    
    data_norm[:2, :, :] = data[:2, :, :] - hip_center
    
    if C >= 3: # If 3D (X, Y, Z)
        z_center = (data[2, :, KP_L_HIP] + data[2, :, KP_R_HIP]) / 2.0
        z_center = np.expand_dims(z_center, axis=-1)
        data_norm[2, :, :] = data[2, :, :] - z_center

    # 2. View Alignment (Rotation to Canonical Frontal View)
    # DISABLE FOR ROBUSTNESS: 
    # Normalizing rotation counteracts 3D augmentation (back view sim, rotation jitter).
    # We want the model to learn invariance directly from the augmented data.
    if False and C >= 3: 
        # Vector between hips
        left_hip = data_norm[:3, :, KP_L_HIP]
        right_hip = data_norm[:3, :, KP_R_HIP]
        hip_vec = left_hip - right_hip # (3, T)
        
        # We want hip_vec to align with X-axis (1, 0, 0)
        # Calculate angle in XZ plane to rotate around Y-axis
        # atan2(z, x)
        angles = np.arctan2(hip_vec[2, :], hip_vec[0, :]) # (T,)
        
        # Create rotation matrices for each frame to cancel out the angle
        # Rotate by -angle to align with X-axis
        cos_a = np.cos(-angles)
        sin_a = np.sin(-angles)
        
        # R_y matrix:
        # [ cos  0  sin]
        # [  0   1   0 ]
        # [-sin  0  cos]
        
        # Simple application per frame
        x_new = data_norm[0] * cos_a[:, np.newaxis] + data_norm[2] * sin_a[:, np.newaxis]
        z_new = -data_norm[0] * sin_a[:, np.newaxis] + data_norm[2] * cos_a[:, np.newaxis]
        
        data_norm[0] = x_new
        data_norm[2] = z_new

    # 3. Scaling (Torso Size Invariance)
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

    data_norm[:3, :, :] /= mean_torso
    
    # Robustness: Clip outliers, especially from unstable 3D Z-estimation
    # Normalized data typically sits in [-1, 1] or [-2, 2] range for body parts.
    # Outliers (e.g. Z=50) can freeze the classifier.
    data_norm[:3, :, :] = np.clip(data_norm[:3, :, :], -5.0, 5.0)

    return data_norm.astype(np.float32)
