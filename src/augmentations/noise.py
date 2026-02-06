"""
Noise and robustness augmentations for skeleton data.
"""
import numpy as np


def local_joint_jittering(data, noise_std=0.01):
    """
    Add independent Gaussian noise to each joint.
    
    Args:
        data: (C, T, V)
        noise_std: noise standard deviation
    
    Returns:
        Jittered skeleton data
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
    
    Returns:
        Skeleton with scaled bone lengths
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
    
    Returns:
        Skeleton with masked confidence values
    """
    C, T, V = data.shape
    
    if C < 3:  # Need confidence channel
        return data
    
    mask = np.random.binomial(1, mask_prob, (T, V))
    data[2:, :, :] = data[2:, :, :] * (1 - mask[np.newaxis, :, :])
    
    return data.astype(np.float32)


def confidence_jittering(data, low=0.3, high=0.7, prob=0.2):
    """
    Simulates uncertain tracking by randomly reducing confidence 
    of some joints to medium values.
    
    Args:
        data: (C, T, V)
        low: minimum jittered confidence value
        high: maximum jittered confidence value
        prob: probability of jittering each keypoint
    
    Returns:
        Skeleton with jittered confidence values
    """
    C, T, V = data.shape
    if C < 3: 
        return data
    
    # Create a mask to jitter confidence values with a certain probability
    mask = np.random.random((T, V)) < prob
    
    # Compute random confidence values
    noise = np.random.uniform(low, high, (T, V))
    
    # Apply the mask
    data[2, mask] = noise[mask]
    
    return data.astype(np.float32)


def keypoint_dropout(data, dropout_prob=0.05):
    """
    Set random keypoints to zero (simulate missing detections).
    
    Args:
        data: (C, T, V)
        dropout_prob: probability of dropping each keypoint
    
    Returns:
        Skeleton with dropped keypoints
    """
    C, T, V = data.shape
    dropout_mask = np.random.binomial(1, 1 - dropout_prob, (T, V))
    
    data[:2, :, :] = data[:2, :, :] * dropout_mask[np.newaxis, :, :]
    
    if C > 2:
        data[2:, :, :] = data[2:, :, :] * dropout_mask[np.newaxis, :, :]
    
    return data.astype(np.float32)
