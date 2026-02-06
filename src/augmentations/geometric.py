"""
Geometric augmentations for skeleton data.
"""
import numpy as np
from ..constants import COCO_SWAP_PAIRS, COCO_UPPER_BODY, COCO_LOWER_BODY


def flip_horizontal(data):
    """
    Horizontal flip (Right handed <-> Left handed).
    Flips X coordinate and swaps left/right keypoints.
    
    Args:
        data: (C, T, V) skeleton data
    
    Returns:
        Horizontally flipped data
    """
    C, T, V = data.shape
    data[0, :, :] = -data[0, :, :]
    
    for left, right in COCO_SWAP_PAIRS:
        if left < V and right < V:
            temp = data[:, :, left].copy()
            data[:, :, left] = data[:, :, right]
            data[:, :, right] = temp
    
    return data


def global_rotation(data, rotation_range=10):
    """
    Global rotation around Z axis (camera view).
    
    Args:
        data: (C, T, V) skeleton data
        rotation_range: rotation range in degrees
    
    Returns:
        Rotated skeleton data
    """
    C, T, V = data.shape
    
    if C < 2:
        return data
    
    theta = np.radians(np.random.uniform(-rotation_range, rotation_range))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    
    data_xy = data[:2, :, :]
    xy_reshaped = data_xy.reshape(2, -1)
    xy_rotated = np.dot(rotation_matrix, xy_reshaped)
    data[:2, :, :] = xy_rotated.reshape(2, T, V)
    
    return data


def global_scaling(data, scale_range=0.1):
    """
    Global scaling (simulated zoom).
    
    Args:
        data: (C, T, V) skeleton data
        scale_range: scaling range
    
    Returns:
        Scaled skeleton data
    """
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    data[:2, :, :] = data[:2, :, :] * scale
    return data


def shearing(data, shear_range=0.15):
    """
    Shearing transformation in XY plane.
    
    Args:
        data: (C, T, V)
        shear_range: shear intensity
    
    Returns:
        Sheared skeleton data
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


def pose_rotation(data, rotation_range=15, center_idx=0):
    """
    Rotate pose around different body centers (not just Z axis).
    
    Args:
        data: (C, T, V)
        rotation_range: rotation range in degrees
        center_idx: which keypoint to use as center (0=nose for COCO)
    
    Returns:
        Rotated skeleton data
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
    
    Returns:
        Zoomed skeleton data
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
    
    # Calculate center of selected joints
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
