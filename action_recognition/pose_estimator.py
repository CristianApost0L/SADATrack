import sys
sys.path.append('../')

import constants
import numpy as np

from utils import (
    crop_players,
    smooth_keypoints
)

def estimate_poses(pose_estimator, player_detections, enhanced_frames, last_known_positions):
    '''
    Runs pose estimation on detected players, smooths the keypoints, and resolves 
    logical player identities (Player 1 vs Player 2).

    This function performs the following steps:
    1. Crops player images from the enhanced frames based on bounding boxes.
    2. Runs the YOLO Pose model in batches for efficiency.
    3. Maps keypoints back to the original video coordinate space.
    4. Smooths keypoints over time to reduce jitter and fill missing frames.
    5. Maps temporary Track IDs to logical Player IDs (1 = Bottom/Close, 2 = Top/Far) 
       using spatial heuristics and history from previous clips.
    '''
    
    # 1. Collect all crops from the entire video first
    all_crops = []
    crop_metadata = [] # Stores (frame_idx, track_id, crop_x1, crop_y1) to map back later
    
    # crops the players out of videos and returns those crops
    all_crops, crop_metadata = crop_players(player_detections=player_detections, enhanced_frames=enhanced_frames)

    # 2. Run Inference in Batches (Much Faster)
    # We process 24 crops at a time to saturate the GPU without running out of memory
    BATCH_SIZE = constants.BATCH_SIZE
    all_pose_results = []
    
    # Process using a progress bar if you want, or just a range loop
    for i in range(0, len(all_crops), BATCH_SIZE):
        batch_crops = all_crops[i : i + BATCH_SIZE]
        
        # verbose=False prevents it from printing 1000s of lines
        # stream=False ensures we get a list of results back immediately
        batch_results = pose_estimator(batch_crops, verbose=False, stream=False)
        all_pose_results.extend(batch_results)

    # 3. Map Results Back to Player Detections
    for i, result in enumerate(all_pose_results):
        frame_idx, track_id, crop_x1, crop_y1 = crop_metadata[i]
        
        found_keypoints = False
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            # Extract keypoints (17, 3)
            kpts = result.keypoints.data[0].cpu().numpy()
            
            # Add the crop offset to map back to original frame coordinates
            kpts[:, 0] += crop_x1
            kpts[:, 1] += crop_y1
            
            player_detections[frame_idx][track_id]['keypoints'] = kpts.tolist()
            found_keypoints = True
        
        if not found_keypoints:
            player_detections[frame_idx][track_id]['keypoints'] = []

    # KEYPOINT SMOOTHING 
    # Doing it here updates 'player_detections' IN PLACE.
    # Both the HDGCN model AND the Drawing loop will see stable skeletons.
    print("Smoothing skeleton keypoints...")
    player_detections = smooth_keypoints(player_detections)

    # --- DYNAMIC ID MAPPING (IMPROVED) ---
    # Goal: Robustly identify Player 1 (Closest/Bottom) and Player 2 (Farthest/Top)
    
    # 1. Filter Noise: Find the two most frequent Track IDs
    id_occupancy = {}
    for frame_dict in player_detections:
        for track_id in frame_dict.keys():
            id_occupancy[track_id] = id_occupancy.get(track_id, 0) + 1
            
    # Select top 2 IDs based on how many frames they appear in
    # This removes ball boys or line judges who are only detected briefly
    valid_ids = sorted(id_occupancy, key=id_occupancy.get, reverse=True)[:2]
    
    # 2. Assign IDs based on "Size" (Bounding Box Height)
    # The closer player (Player 1) will have a larger bounding box height.
    id_avg_height = {}
    
    for pid in valid_ids:
        heights = []
        for frame_dict in player_detections:
            if pid in frame_dict:
                bbox = frame_dict[pid]['bbox']
                # Calculate height: y2 - y1
                h = bbox[3] - bbox[1]
                heights.append(h)
        
        # Calculate average height for this player ID
        if heights:
            id_avg_height[pid] = sum(heights) / len(heights)
        else:
            id_avg_height[pid] = 0

    # Sort IDs by Height: Largest (Closest) -> Smallest (Farthest)
    sorted_ids = sorted(valid_ids, key=lambda x: id_avg_height.get(x, 0), reverse=True)
    
    player_id_map = {}
    
    # --- NEW LOGIC: Position-Based Handover ---
    mapped_via_position = False
    
    if last_known_positions is not None and len(valid_ids) >= 1:
        print("Attempting to map IDs based on previous clip positions...")
        
        # Get centroids of the current valid IDs in the FIRST frame they appear
        current_centroids = {}
        for pid in valid_ids:
            # Find first frame where this pid appears
            for frame_dict in player_detections:
                if pid in frame_dict:
                    bbox = frame_dict[pid]['bbox']
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    current_centroids[pid] = np.array([cx, cy])
                    break
        
        # Match current IDs to P1/P2 from previous clip
        used_pids = set()
        
        for p_num, prev_pos in last_known_positions.items():
            best_pid = None
            min_dist = float('inf')
            prev_pos_arr = np.array(prev_pos)
            
            for pid, curr_pos in current_centroids.items():
                if pid in used_pids: continue
                
                dist = np.linalg.norm(curr_pos - prev_pos_arr)
                # Threshold: Players shouldn't jump > 300px between clips
                if dist < min_dist and dist < 300: 
                    min_dist = dist
                    best_pid = pid
            
            if best_pid is not None:
                player_id_map[best_pid] = p_num
                used_pids.add(best_pid)
                print(f"Mapped Track ID {best_pid} -> Player {p_num} (Dist: {min_dist:.1f}px)")
                mapped_via_position = True

    # --- FALLBACK: ORIGINAL HEIGHT HEURISTIC ---
    # If we couldn't map via position (first clip, or tracking lost), use height
    if not mapped_via_position or len(player_id_map) < len(valid_ids):
        print("Using Height Heuristic for remaining IDs...")
        
        # Assign Player 1 (Closest/Largest) if not already assigned
        if len(sorted_ids) >= 1:
            p1_candidate = sorted_ids[0]
            if 1 not in player_id_map.values() and p1_candidate not in player_id_map:
                player_id_map[p1_candidate] = 1
                print(f"Identified Player 1 (Closest): Track ID {p1_candidate}")
            
        # Assign Player 2 (Farthest/Smallest) if not already assigned
        if len(sorted_ids) >= 2:
            p2_candidate = sorted_ids[1]
            # Special check: If p2_candidate was assigned to P1 (because P1 is missing), don't overwrite
            if 2 not in player_id_map.values() and p2_candidate not in player_id_map:
                player_id_map[p2_candidate] = 2
                print(f"Identified Player 2 (Farthest): Track ID {p2_candidate}")
    
    # Fallback: Map any other stray IDs to Player 1 to prevent crashes
    all_detected_ids = set(id_occupancy.keys())
    for pid in all_detected_ids:
        if pid not in player_id_map:
            player_id_map[pid] = 1

    return player_id_map, player_detections