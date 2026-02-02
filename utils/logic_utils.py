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

# Helper function to measure distance from ball to nearest player
def get_proximity_score(ball_detections, player_detections, frame_idx):
    ball_pos = ball_detections[frame_idx]
    if ball_pos is None: return float('inf')
    bx, by = ball_pos
    
    min_dist = float('inf')
    # Check against all players detected in this frame
    if frame_idx < len(player_detections):
        for pid, p_data in player_detections[frame_idx].items():
            bbox = p_data['bbox']
            # Player Center
            px = (bbox[0] + bbox[2]) / 2
            py = (bbox[1] + bbox[3]) / 2
            dist = ((bx - px)**2 + (by - py)**2)**0.5
            if dist < min_dist: min_dist = dist
    return min_dist