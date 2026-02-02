def filter_adjacent_frames(frame_list, min_distance=24):
    """
    If multiple shots are detected within 'min_distance' frames (e.g. 1 sec),
    keep only the first one.
    """
    if not frame_list:
        return []
        
    sorted_frames = sorted(frame_list)
    filtered_frames = [sorted_frames[0]]
    
    for frame in sorted_frames[1:]:
        # Only accept the frame if it is at least 'min_distance' away from the last one
        if frame - filtered_frames[-1] > min_distance:
            filtered_frames.append(frame)
            
    return filtered_frames