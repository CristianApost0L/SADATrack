"""
PyTorch Dataset for Tennis Swing Classification.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

# Import from modular structure
from .constants import THETIS_CLASSES, LABEL_MAP, COCO_BONE_PAIRS
from .normalization import normalize_skeleton
from .augmentations import augment_skeleton
from .curriculum import CurriculumLearningScheduler
from .utils import get_thetis_files

# Re-export for backward compatibility
__all__ = [
    'TennisDataset',
    'CurriculumLearningScheduler',
    'get_thetis_files',
    'THETIS_CLASSES',
    'LABEL_MAP',
]


class TennisDataset(Dataset):
    """
    PyTorch Dataset for Tennis Swing Classification.
    
    Supports:
    - Data augmentation with comprehensive pipeline
    - Bone representation (vectors between joints)
    - Forced augmentations for validation (back view, flip)
    """
    
    def __init__(self, X, y, augment=False, augmentation_probs=None, data_type='joint', 
                 force_back_view_val=False, force_flip_val=False):
        """
        Args:
            X: numpy array (N, T, V, C) from prepare_data.py
            y: numpy array (N,) labels
            augment: bool, if True applies random transformations
            augmentation_probs: dict with probabilities for each augmentation type
            data_type: 'joint' or 'bone'. If 'bone', converts joints to vectors.
            force_back_view_val: bool, if True forces back view simulation (for robust validation)
            force_flip_val: bool, if True forces horizontal flip (for handedness validation)
        """
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2) 
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.data_type = data_type
        self.force_back_view_val = force_back_view_val
        self.force_flip_val = force_flip_val
        
        # Default augmentation probabilities
        self.aug_probs = {
            'flip_prob': 0.5,
            'rotation_range': 10,
            'scale_range': 0.1,
            'noise_std': 0.005,
            'apply_temporal_crop': True,
            'apply_temporal_scaling': True,
            'apply_frame_dropping': True,
            'apply_shearing': True,
            'apply_local_jitter': True,
            'apply_bone_scaling': True,
            'apply_confidence_mask': True,
            'apply_keypoint_dropout': True,
            'apply_pose_rotation': True,
            'apply_local_zoom': True,
            'apply_confidence_jitter': True,
            'apply_back_view': True,
            'force_back_view': False
        }
        
        # Update with user-provided probabilities
        if augmentation_probs:
            self.aug_probs.update(augmentation_probs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx].clone().numpy() 
        label = self.y[idx]
        
        # Special case: Forced Back View for Validation
        if self.force_back_view_val:
            sample = augment_skeleton(
                sample, 
                apply_back_view=True, 
                force_back_view=True, 
                # Disable everything else
                flip_prob=0, rotation_range=0, scale_range=0, noise_std=0,
                apply_temporal_crop=False, apply_temporal_scaling=False,
                apply_frame_dropping=False, apply_shearing=False, 
                apply_local_jitter=False, apply_bone_scaling=False,
                apply_confidence_mask=False, apply_keypoint_dropout=False,
                apply_pose_rotation=False, apply_local_zoom=False,
                apply_confidence_jitter=False
            )
        
        # Special case: Forced Horizontal Flip for Handedness Validation
        elif self.force_flip_val:
            sample = augment_skeleton(
                sample, 
                force_flip=True,
                # Disable everything else
                flip_prob=0, rotation_range=0, scale_range=0, noise_std=0,
                apply_temporal_crop=False, apply_temporal_scaling=False,
                apply_frame_dropping=False, apply_shearing=False, 
                apply_local_jitter=False, apply_bone_scaling=False,
                apply_confidence_mask=False, apply_keypoint_dropout=False,
                apply_pose_rotation=False, apply_local_zoom=False,
                apply_confidence_jitter=False, apply_back_view=False
            )

        elif self.augment:
            sample = augment_skeleton(sample, **self.aug_probs)
        
        # Always normalize geometry for robust classification
        sample = normalize_skeleton(sample)

        # Convert to bone representation if requested
        if self.data_type == 'bone':
            bone_sample = np.zeros_like(sample)
            for child, parent in COCO_BONE_PAIRS:
                # Vector = Child - Parent (for X, Y usually channels 0, 1)
                bone_sample[:2, :, child] = sample[:2, :, child] - sample[:2, :, parent]
                
                # If there is a confidence channel (3rd channel), keep the child's confidence
                if sample.shape[0] > 2:
                     bone_sample[2:, :, child] = sample[2:, :, child]
            sample = bone_sample

        return torch.FloatTensor(sample), label
    
    def disable_augmentation(self, aug_names):
        """
        Disable specific augmentations.
        
        Args:
            aug_names: str or list of augmentation names to disable
        """
        if isinstance(aug_names, str):
            aug_names = [aug_names]
        
        for name in aug_names:
            if f'apply_{name}' in self.aug_probs:
                self.aug_probs[f'apply_{name}'] = False
    
    def set_augmentation_strength(self, strength='weak'):
        """
        Set overall augmentation strength.
        
        Args:
            strength: 'weak', 'medium', 'strong'
        """
        if strength == 'weak':
            self.aug_probs.update({
                'flip_prob': 0.3,
                'rotation_range': 10,
                'scale_range': 0.05,
                'noise_std': 0.003,
            })
        elif strength == 'medium':
            self.aug_probs.update({
                'flip_prob': 0.5,
                'rotation_range': 20,
                'scale_range': 0.1,
                'noise_std': 0.005,
            })
        elif strength == 'strong':
            self.aug_probs.update({
                'flip_prob': 0.7,
                'rotation_range': 35,
                'scale_range': 0.2,
                'noise_std': 0.01,
            })
    
    def update_augmentation_params(self, augmentation_probs):
        """
        Update augmentation parameters (e.g., from curriculum learning).
        
        Args:
            augmentation_probs: dict with new augmentation parameters
        """
        self.aug_probs.update(augmentation_probs)
