def smooth_keypoints(player_detections, window_length=3):
    """
    Applies a moving average filter to keypoints to reduce jitter.
    """
    from collections import defaultdict
    import numpy as np

    # 1. Organize data by Track ID -> List of frames
    track_history = defaultdict(list)
    
    # Collect all data
    for frame_idx, frame_data in enumerate(player_detections):
        for track_id, data in frame_data.items():
            kpts = np.array(data['keypoints'])
            if len(kpts) > 0:
                track_history[track_id].append((frame_idx, kpts))

    # 2. Apply Smoothing
    for track_id, history in track_history.items():
        frames, kpts_data = zip(*history)
        kpts_data = np.array(kpts_data) # Shape: (N_frames, 17, 3)

        for k in range(17): # For each body part
            for c in range(2): # For x and y
                # Apply moving average
                kpts_data[:, k, c] = np.convolve(kpts_data[:, k, c], np.ones(window_length)/window_length, mode='same')

        # 3. Write back
        for i, frame_idx in enumerate(frames):
            player_detections[frame_idx][track_id]['keypoints'] = kpts_data[i].tolist()

    return player_detections