import os
import numpy as np
import torch
from torch.utils.data import Dataset

# The 12 official TheTis classes
THETIS_CLASSES = [
    "backhand2hands",
    "backhand", 
    "backhand_slice",
    "backhand_volley",
    "forehand_flat",
    "forehand_openstands",
    "forehand_slice",
    "forehand_volley",
    "flat_service",
    "kick_service",
    "slice_service",
    "smash"
]

# Mapping from class name to label index
LABEL_MAP = {cls_name: i for i, cls_name in enumerate(THETIS_CLASSES)}

def _normalize_string(s):
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

def get_thetis_files(root_dir):

    samples = []
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    normalized_classes = {_normalize_string(k): k for k in THETIS_CLASSES}
    
    try:
        subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    except NotADirectoryError:
        print(f"ERROR: {root_dir} does not appear to be a directory.")
        return [], LABEL_MAP

    mapped_folders = 0
    
    for folder_name in subfolders:
        clean_folder = _normalize_string(folder_name)
        official_name = None
        
        if clean_folder in normalized_classes:
            official_name = normalized_classes[clean_folder]
        else:
            for norm_cls, real_cls in normalized_classes.items():
                if norm_cls in clean_folder:
                    official_name = real_cls
                    break
        
        if official_name:
            mapped_folders += 1
            class_id = LABEL_MAP[official_name]
            folder_path = os.path.join(root_dir, folder_name)
            
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.avi', '.mp4', '.mov'))]
            
            for video_file in files:
                full_path = os.path.join(folder_path, video_file)
                samples.append((full_path, class_id))
                
            print(f"  [OK] Directory '{folder_name}' mapped with '{official_name}' with {len(files)} videos.")
        else:
            print(f"  [SKIP] Directory '{folder_name}' could not be mapped to any official class.")

    return samples, LABEL_MAP

# COCO Keypoints Left/Right pairs
COCO_SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# COCO Keypoint indices for body parts (for local zoom)
COCO_UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # head, shoulders, arms
COCO_LOWER_BODY = [11, 12, 13, 14, 15, 16]  # hips, legs


def temporal_crop(data, crop_ratio=0.8):
    """
    Random temporal cropping - extract a random segment of the sequence.
    Args:
        data: (C, T, V)
        crop_ratio: percentage of frames to keep (0.8 = keep 80%)
    """
    C, T, V = data.shape
    crop_length = max(1, int(T * crop_ratio))
    start_idx = np.random.randint(0, T - crop_length + 1)
    
    cropped = data[:, start_idx:start_idx + crop_length, :]
    
    # Pad back to original length if needed
    if crop_length < T:
        pad_length = T - crop_length
        pad_before = np.random.randint(0, pad_length + 1)
        pad_after = pad_length - pad_before
        
        # Repeat first and last frames for padding
        first_frame = cropped[:, [0], :]
        last_frame = cropped[:, [-1], :]
        
        if pad_before > 0:
            cropped = np.concatenate([np.repeat(first_frame, pad_before, axis=1), cropped], axis=1)
        if pad_after > 0:
            cropped = np.concatenate([cropped, np.repeat(last_frame, pad_after, axis=1)], axis=1)
    
    return cropped


def temporal_scaling(data, scale_range=(0.8, 1.2)):
    """
    Temporal scaling - compress or expand the sequence (speed up/slow down).
    Args:
        data: (C, T, V)
        scale_range: (min_scale, max_scale)
    """
    C, T, V = data.shape
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_length = max(1, int(T * scale))
    
    # Linear interpolation in time
    indices = np.linspace(0, T - 1, new_length)
    scaled = np.zeros((C, new_length, V), dtype=np.float32)
    
    for c in range(C):
        for v in range(V):
            scaled[c, :, v] = np.interp(indices, np.arange(T), data[c, :, v])
    
    # Pad/crop to original length
    if new_length < T:
        pad_length = T - new_length
        pad_before = np.random.randint(0, pad_length + 1)
        pad_after = pad_length - pad_before
        last_frame = scaled[:, [-1], :]
        first_frame = scaled[:, [0], :]
        if pad_before > 0:
            scaled = np.concatenate([np.repeat(first_frame, pad_before, axis=1), scaled], axis=1)
        if pad_after > 0:
            scaled = np.concatenate([scaled, np.repeat(last_frame, pad_after, axis=1)], axis=1)
    elif new_length > T:
        scaled = scaled[:, :T, :]
    
    return scaled


def frame_dropping(data, drop_prob=0.1):
    """
    Random frame dropping - simulate missing frames.
    Args:
        data: (C, T, V)
        drop_prob: probability of dropping each frame
    """
    C, T, V = data.shape
    keep_mask = np.random.binomial(1, 1 - drop_prob, T).astype(bool)
    
    # Keep at least 1 frame
    if keep_mask.sum() == 0:
        keep_mask[np.random.randint(0, T)] = True
    
    kept_data = data[:, keep_mask, :]
    
    # Interpolate back to original length
    kept_indices = np.where(keep_mask)[0]
    indices = np.linspace(0, len(kept_indices) - 1, T)
    
    result = np.zeros((C, T, V), dtype=np.float32)
    for c in range(C):
        for v in range(V):
            result[c, :, v] = np.interp(indices, np.arange(len(kept_indices)), kept_data[c, :, v])
    
    return result


def shearing(data, shear_range=0.15):
    """
    Shearing transformation in XY plane.
    Args:
        data: (C, T, V)
        shear_range: shear intensity
    """
    C, T, V = data.shape
    
    if C < 2:
        return data
    
    data_xy = data[:2, :, :].copy()
    data_rest = data[2:, :, :] if C > 2 else None
    
    # Random shear matrix: [[1, shx], [shy, 1]]
    shx = np.random.uniform(-shear_range, shear_range)
    shy = np.random.uniform(-shear_range, shear_range)
    shear_matrix = np.array([[1, shx], [shy, 1]])
    
    # Apply shearing
    xy_reshaped = data_xy.reshape(2, -1)
    xy_sheared = np.dot(shear_matrix, xy_reshaped)
    data_xy = xy_sheared.reshape(2, T, V)
    
    if data_rest is not None:
        return np.concatenate([data_xy, data_rest], axis=0).astype(np.float32)
    return data_xy.astype(np.float32)


def local_joint_jittering(data, noise_std=0.01):
    """
    Add independent Gaussian noise to each joint.
    Args:
        data: (C, T, V)
        noise_std: noise standard deviation
    """
    C, T, V = data.shape
    noise = np.random.normal(0, noise_std, (C, T, V))
    return (data + noise).astype(np.float32)


def bone_length_scaling(data, scale_range=0.1):
    """
    Scale bone lengths (distances between connected joints).
    Assumes COCO skeleton connections.
    Args:
        data: (C, T, V)
        scale_range: scaling intensity
    """
    C, T, V = data.shape
    
    if C < 2:
        return data
    
    data_xy = data[:2, :, :].copy()
    data_rest = data[2:, :, :] if C > 2 else None
    
    # COCO skeleton edges
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head/shoulders
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]
    
    for start, end in edges:
        if start < V and end < V:
            scale = np.random.uniform(1 - scale_range, 1 + scale_range)
            
            # Get center point
            center = (data_xy[:, :, start] + data_xy[:, :, end]) / 2
            
            # Scale distance from center
            data_xy[:, :, start] = center + (data_xy[:, :, start] - center) * scale
            data_xy[:, :, end] = center + (data_xy[:, :, end] - center) * scale
    
    if data_rest is not None:
        return np.concatenate([data_xy, data_rest], axis=0).astype(np.float32)
    return data_xy.astype(np.float32)


def confidence_masking(data, mask_prob=0.1):
    """
    Reduce confidence of random keypoints to simulate tracking loss.
    Args:
        data: (C, T, V)
        mask_prob: probability of masking each keypoint
    """
    C, T, V = data.shape
    
    if C < 3:  # Need confidence channel
        return data
    
    mask = np.random.binomial(1, mask_prob, (T, V))
    data[2:, :, :] = data[2:, :, :] * (1 - mask[np.newaxis, :, :])
    
    return data.astype(np.float32)


def keypoint_dropout(data, dropout_prob=0.05):
    """
    Set random keypoints to zero (simulate missing detections).
    Args:
        data: (C, T, V)
        dropout_prob: probability of dropping each keypoint
    """
    C, T, V = data.shape
    dropout_mask = np.random.binomial(1, 1 - dropout_prob, (T, V))
    
    data[:2, :, :] = data[:2, :, :] * dropout_mask[np.newaxis, :, :]
    
    if C > 2:
        data[2:, :, :] = data[2:, :, :] * dropout_mask[np.newaxis, :, :]
    
    return data.astype(np.float32)


def pose_rotation(data, rotation_range=15, center_idx=0):
    """
    Rotate pose around different body centers (not just Z axis).
    Args:
        data: (C, T, V)
        rotation_range: rotation range in degrees
        center_idx: which keypoint to use as center (0=nose for COCO)
    """
    C, T, V = data.shape
    
    if C < 2:
        return data
    
    data_xy = data[:2, :, :].copy()
    data_rest = data[2:, :, :] if C > 2 else None
    
    # Random rotation center
    if center_idx < V:
        center_x = data_xy[0, :, center_idx]
        center_y = data_xy[1, :, center_idx]
    else:
        center_x = np.mean(data_xy[0, :, :], axis=1)
        center_y = np.mean(data_xy[1, :, :], axis=1)
    
    theta = np.radians(np.random.uniform(-rotation_range, rotation_range))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    
    # Apply rotation per frame
    for t in range(T):
        # Center translation
        xy_centered = data_xy[:, t, :] - np.array([[center_x[t]], [center_y[t]]])
        # Rotation
        xy_rotated = np.dot(rotation_matrix, xy_centered)
        # Back translation
        data_xy[:, t, :] = xy_rotated + np.array([[center_x[t]], [center_y[t]]])
    
    if data_rest is not None:
        return np.concatenate([data_xy, data_rest], axis=0).astype(np.float32)
    return data_xy.astype(np.float32)


def local_zoom(data, zoom_range=(0.8, 1.2), body_part='random'):
    """
    Zoom on different body parts independently.
    Args:
        data: (C, T, V)
        zoom_range: (min_zoom, max_zoom)
        body_part: 'random', 'upper', 'lower', 'left', 'right'
    """
    C, T, V = data.shape
    
    if C < 2:
        return data
    
    data_xy = data[:2, :, :].copy()
    data_rest = data[2:, :, :] if C > 2 else None
    
    # Select which joints to zoom
    if body_part == 'upper':
        joints = COCO_UPPER_BODY
    elif body_part == 'lower':
        joints = COCO_LOWER_BODY
    elif body_part == 'left':
        joints = [i for i in range(V) if i % 2 == 1]
    elif body_part == 'right':
        joints = [i for i in range(V) if i % 2 == 0]
    else:  # random
        body_part = np.random.choice(['upper', 'lower'])
        joints = COCO_UPPER_BODY if body_part == 'upper' else COCO_LOWER_BODY
    
    joints = [j for j in joints if j < V]
    
    if len(joints) == 0:
        return data
    
    zoom = np.random.uniform(zoom_range[0], zoom_range[1])
    
    # Calculate center of selected joints - average across selected joints for each frame
    # Extract joint coordinates for selected joints
    selected_x = data_xy[0, :, :][:, joints]  # shape: (T, num_joints)
    selected_y = data_xy[1, :, :][:, joints]  # shape: (T, num_joints)
    
    center_x = np.mean(selected_x, axis=1, keepdims=True)  # shape: (T, 1)
    center_y = np.mean(selected_y, axis=1, keepdims=True)  # shape: (T, 1)
    
    # Apply zoom to each selected joint
    for j in joints:
        data_xy[0, :, j] = center_x[:, 0] + (data_xy[0, :, j] - center_x[:, 0]) * zoom
        data_xy[1, :, j] = center_y[:, 0] + (data_xy[1, :, j] - center_y[:, 0]) * zoom
    
    if data_rest is not None:
        return np.concatenate([data_xy, data_rest], axis=0).astype(np.float32)
    return data_xy.astype(np.float32)


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
                     apply_local_zoom=True):
    """
    Comprehensive skeleton augmentation pipeline.
    
    Args:
        data: (C, T, V) -> (Channels, Time, Vertices)
        flip_prob: probability of horizontal flip (0.5 = 50%)
        rotation_range: Z-axis rotation range in degrees
        scale_range: global scale range
        noise_std: Gaussian noise std
        apply_*: whether to apply each augmentation
    """
    C, T, V = data.shape
    
    if C < 2:
        return data
    
    # Temporal augmentations (should be applied first)
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
    if np.random.random() < flip_prob:
        data[0, :, :] = -data[0, :, :]
        for left, right in COCO_SWAP_PAIRS:
            if left < V and right < V:
                temp = data[:, :, left].copy()
                data[:, :, left] = data[:, :, right]
                data[:, :, right] = temp
    
    # 2. Global Rotation (around Z axis / camera view)
    if np.random.random() < 0.5:
        theta = np.radians(np.random.uniform(-rotation_range, rotation_range))
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        data_xy = data[:2, :, :]
        xy_reshaped = data_xy.reshape(2, -1)
        xy_rotated = np.dot(rotation_matrix, xy_reshaped)
        data[:2, :, :] = xy_rotated.reshape(2, T, V)
    
    # 3. Global Scaling (Simulated zoom)
    if np.random.random() < 0.5:
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        data[:2, :, :] = data[:2, :, :] * scale
    
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
    
    if apply_keypoint_dropout and np.random.random() < 0.2:
        data = keypoint_dropout(data)
    
    return data.astype(np.float32)


class CurriculumLearningScheduler:
    """
    Gradually increase augmentation strength during training.
    Follows a curriculum: weak -> medium -> strong
    """
    def __init__(self, total_epochs, schedule_type='linear'):
        """
        Args:
            total_epochs: total number of training epochs
            schedule_type: 'linear', 'exponential', 'step'
        """
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        
        # Define augmentation parameters at each stage
        self.stages = {
            'weak': {
                'flip_prob': 0.3,
                'rotation_range': 10,
                'scale_range': 0.05,
                'noise_std': 0.003,
            },
            'medium': {
                'flip_prob': 0.5,
                'rotation_range': 20,
                'scale_range': 0.1,
                'noise_std': 0.005,
            },
            'strong': {
                'flip_prob': 0.7,
                'rotation_range': 35,
                'scale_range': 0.2,
                'noise_std': 0.01,
            }
        }
    
    def get_augmentation_params(self, current_epoch):
        """
        Get augmentation parameters for current epoch.
        Args:
            current_epoch: current epoch number (0-indexed)
        Returns:
            dict with augmentation parameters
        """
        progress = current_epoch / self.total_epochs
        
        if self.schedule_type == 'linear':
            return self._linear_interpolate(progress)
        elif self.schedule_type == 'exponential':
            return self._exponential_interpolate(progress)
        elif self.schedule_type == 'step':
            return self._step_schedule(progress)
        else:
            return self.stages['weak']
    
    def _linear_interpolate(self, progress):
        """Linear interpolation between weak and strong."""
        if progress < 0.5:
            # Weak to Medium (0 to 0.5)
            alpha = progress * 2  # 0 to 1
            return self._interpolate_dicts(self.stages['weak'], self.stages['medium'], alpha)
        else:
            # Medium to Strong (0.5 to 1)
            alpha = (progress - 0.5) * 2  # 0 to 1
            return self._interpolate_dicts(self.stages['medium'], self.stages['strong'], alpha)
    
    def _exponential_interpolate(self, progress):
        """Exponential interpolation - slower start, faster end."""
        # Exponential ease-in: faster progression toward the end
        progress_exp = progress ** 0.5  # Square root for smoother start
        
        if progress_exp < 0.5:
            alpha = progress_exp * 2
            return self._interpolate_dicts(self.stages['weak'], self.stages['medium'], alpha)
        else:
            alpha = (progress_exp - 0.5) * 2
            return self._interpolate_dicts(self.stages['medium'], self.stages['strong'], alpha)
    
    def _step_schedule(self, progress):
        """Step schedule - abrupt transitions."""
        if progress < 0.33:
            return self.stages['weak'].copy()
        elif progress < 0.66:
            return self.stages['medium'].copy()
        else:
            return self.stages['strong'].copy()
    
    def _interpolate_dicts(self, dict1, dict2, alpha):
        """Linear interpolation between two parameter dicts."""
        result = {}
        for key in dict1:
            result[key] = dict1[key] + (dict2[key] - dict1[key]) * alpha
        return result


class TennisDataset(Dataset):
    def __init__(self, X, y, augment=False, augmentation_probs=None):
        """
        Args:
            X: numpy array (N, T, V, C) from prepare_data.py
            y: numpy array (N,) labels
            augment: bool, if True applies random transformations
            augmentation_probs: dict with probabilities for each augmentation type
                Example: {
                    'flip_prob': 0.5,
                    'rotation_range': 10,
                    'apply_temporal_crop': True,
                    'apply_bone_scaling': True,
                    ...
                }
        """
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2) 
        self.y = torch.LongTensor(y)
        self.augment = augment
        
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
        }
        
        # Update with user-provided probabilities
        if augmentation_probs:
            self.aug_probs.update(augmentation_probs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx].clone().numpy() 
        label = self.y[idx]
        
        if self.augment:
            sample = augment_skeleton(sample, **self.aug_probs)
            
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