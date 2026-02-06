"""
Comprehensive augmentation pipeline for skeleton sequences.
"""
import numpy as np
from .geometric import flip_horizontal, global_rotation, global_scaling, shearing, pose_rotation, local_zoom
from .temporal import temporal_crop, temporal_scaling, frame_dropping
from .noise import local_joint_jittering, bone_length_scaling, confidence_masking, confidence_jittering, keypoint_dropout
from .view import augment_3d_view_rotation, simulate_back_view


def augment_skeleton(data, 
                     flip_prob=0.5,
                     rotation_range=10, 
                     scale_range=0.1, 
                     noise_std=0.005,
                     apply_temporal_crop=True,
                     apply_temporal_scaling=True,
                     apply_frame_dropping=True,
                     apply_shearing=True,
                     apply_local_jitter=True,
                     apply_bone_scaling=True,
                     apply_confidence_mask=True,
                     apply_keypoint_dropout=True,
                     apply_pose_rotation=True,
                     apply_local_zoom=True,
                     apply_confidence_jitter=True,
                     apply_back_view=True,
                     force_back_view=False,
                     force_flip=False):
    """
    Comprehensive skeleton augmentation pipeline.
    
    Args:
        data: (C, T, V) -> (Channels, Time, Vertices)
        flip_prob: probability of horizontal flip (0.5 = 50%)
        rotation_range: Z-axis rotation range in degrees
        scale_range: global scale range
        noise_std: Gaussian noise std
        apply_*: whether to apply each augmentation
        force_back_view: force back view simulation (for validation)
        force_flip: force horizontal flip (for handedness validation)
    
    Returns:
        Augmented skeleton data (C, T, V) as float32
    """
    C, T, V = data.shape
    
    if C < 2:
        return data

    # 0. Back View Simulation (Solving the Frontal Bias)
    if C >= 2 and (force_back_view or (apply_back_view and np.random.random() < 0.5)):
        data = simulate_back_view(data)

    # 3D View Augmentation (Simulate Camera Angle) - Requires Z coordinate
    if C >= 3 and apply_pose_rotation and np.random.random() < 0.5:
        data = augment_3d_view_rotation(data, angle_range=rotation_range)

    # Temporal augmentations
    if apply_temporal_crop and np.random.random() < 0.3:
        data = temporal_crop(data)
    
    if apply_temporal_scaling and np.random.random() < 0.3:
        data = temporal_scaling(data)
    
    if apply_frame_dropping and np.random.random() < 0.2:
        data = frame_dropping(data)
    
    # Spatial augmentations
    if apply_shearing and np.random.random() < 0.3:
        data = shearing(data)
    
    if apply_pose_rotation and np.random.random() < 0.3:
        data = pose_rotation(data)
    
    if apply_local_zoom and np.random.random() < 0.3:
        data = local_zoom(data)
    
    # 1. Horizontal Flip (Destrorso <-> Mancino)
    if force_flip or np.random.random() < flip_prob:
        data = flip_horizontal(data)
    
    # 2. Global Rotation (around Z axis / camera view)
    if np.random.random() < 0.5:
        data = global_rotation(data, rotation_range)
    
    # 3. Global Scaling (Simulated zoom)
    if np.random.random() < 0.5:
        data = global_scaling(data, scale_range)
    
    # 4. Global Gaussian Noise (Sensor jittering)
    if np.random.random() < 0.5:
        noise = np.random.normal(0, noise_std, data.shape)
        data = data + noise
    
    # Joint-level augmentations
    if apply_local_jitter and np.random.random() < 0.5:
        data = local_joint_jittering(data)
    
    if apply_bone_scaling and np.random.random() < 0.3:
        data = bone_length_scaling(data)
    
    # Confidence-based augmentations
    if apply_confidence_mask and np.random.random() < 0.3:
        data = confidence_masking(data)

    if apply_confidence_jitter and np.random.random() < 0.3:
        data = confidence_jittering(data)
    
    if apply_keypoint_dropout and np.random.random() < 0.2:
        data = keypoint_dropout(data)
    
    return data.astype(np.float32)
