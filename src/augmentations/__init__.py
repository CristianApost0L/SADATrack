"""
Skeleton augmentation modules for action recognition.
"""
from .geometric import flip_horizontal, global_rotation, global_scaling, shearing, pose_rotation, local_zoom
from .temporal import temporal_crop, temporal_scaling, frame_dropping
from .noise import local_joint_jittering, bone_length_scaling, confidence_masking, confidence_jittering, keypoint_dropout
from .view import augment_3d_view_rotation, simulate_back_view
from .pipeline import augment_skeleton

__all__ = [
    # Geometric
    'flip_horizontal',
    'global_rotation',
    'global_scaling',
    'shearing',
    'pose_rotation',
    'local_zoom',
    # Temporal
    'temporal_crop',
    'temporal_scaling',
    'frame_dropping',
    # Noise
    'local_joint_jittering',
    'bone_length_scaling',
    'confidence_masking',
    'confidence_jittering',
    'keypoint_dropout',
    # View
    'augment_3d_view_rotation',
    'simulate_back_view', 
    # Pipeline
    'augment_skeleton',
]
