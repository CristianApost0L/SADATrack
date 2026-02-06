"""
Temporal augmentations for skeleton sequences.
"""
import numpy as np


def temporal_crop(data, crop_ratio=0.8):
    """
    Random temporal cropping - extract a random segment of the sequence.
    
    Args:
        data: (C, T, V)
        crop_ratio: percentage of frames to keep (0.8 = keep 80%)
    
    Returns:
        Temporally cropped sequence (padded back to original length)
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
    
    Returns:
        Temporally scaled sequence (padded/cropped to original length)
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
    
    Returns:
        Sequence with dropped frames (interpolated back to original length)
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
