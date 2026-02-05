from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters,
                   draw_skeletons,
                   enhance_video_contrast,
                   smooth_keypoints,
                   print_validation_report,
                   filter_adjacent_frames,
                   get_proximity_score,
                   split_video_into_clips,
                   merge_clips
                   )
import constants
from trackers import PlayerTracker, BallTracker, BounceDetector 
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
import numpy as np
from copy import deepcopy
from action_recognition.model import HDGCN_Tennis
from action_recognition.extractor import PoseExtractor
import torch
import os
import argparse
import time
import json
import shutil
from ultralytics import YOLO 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def process_single_clip(input_video, output_path, HDGCN_window_size, yolo_verbosity, player_detection_court_margin, original_video_name, frame_offset=0, last_known_positions=None):
    start_time = time.time()

    model_predictions_log = []

    # Read Video
    input_video_path = input_video

    # Initialize Action Classifier
    extractor = PoseExtractor()

    action_model = HDGCN_Tennis(num_classes=12, in_channels=3)
    action_model.load_state_dict(torch.load('/kaggle/input/cv-project/new_Swing_classifier.pth'))
    action_model.eval()
    
    # --- DUAL STREAM SETUP ---
    # Stream A: Raw Frames (Clean, low noise) -> BEST FOR BALL DETECTION
    raw_frames = read_video(input_video_path)
    
    # Stream B: Enhanced Frames (High contrast) -> BEST FOR PLAYER/COURT DETECTION
    print("Preprocessing video for lighting/shadows...")
    enhanced_frames = enhance_video_contrast(raw_frames)


    # Initialize Trackers
    player_tracker = PlayerTracker(model_path='/kaggle/input/cv-project/yolo26x.pt')
    
    ball_tracker = BallTracker(model_path='/kaggle/input/cv-project/ball_model_best.pt') # Amin model

    bounce_detector = BounceDetector(model_path='/kaggle/input/cv-project/ctb_regr_bounce.cbm') # Amin model

    # --- 2. DETECT PLAYERS (Use ENHANCED frames) ---
    print("Detecting Players on Enhanced Video...")
    player_detections = player_tracker.detect_frames(enhanced_frames,
                                                     yolo_verbosity=yolo_verbosity
                                                     )
    
    # FREE MEMORY: We are done with YOLO. Unload it to make room for TrackNet.
    print("Unloading Player Tracker model to free VRAM...")
    del player_tracker.model 
    torch.cuda.empty_cache()

    # --- 3. DETECT BALL (Use RAW frames) ---
    # This ignores the noisy/grainy enhanced frames and looks at the clean original
    print("Detecting Ball on Raw Video...")
    ball_detections = ball_tracker.detect_frames(raw_frames)
    
    # Interpolate ball (standard step)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # 1. Get ALL candidates (Hits + Bounces) using geometric heuristic
    candidate_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    
    # Merge frames like [141, 145, 147] into just [141]
    # 24 frames = 1 second buffer (physically impossible to hit 2 shots in 1 sec)
    candidate_shot_frames = filter_adjacent_frames(candidate_shot_frames, min_distance=24)

    # 2. Detect Bounces using the new Model
    detected_bounces = []
    # Pass the list of (x,y) tuples directly
    detected_bounces = bounce_detector.predict(ball_detections) 

    # 3. Filter: Keep a candidate ONLY if it is NOT a bounce
    clean_candidates = []
    for frame in candidate_shot_frames:
        # Check if this frame is close to any detected bounce (within margin of error, e.g., 3 frames)
        is_bounce = False
        for b_frame in detected_bounces:
            if abs(frame - b_frame) <= 3: 
                is_bounce = True
                break
        
        # If it's not a bounce, it's a hit!
        if not is_bounce:
            clean_candidates.append(frame)

    # SMART FILTERING: Group frames and pick the one CLOSEST to a player
    # Instead of just taking the first frame (filter_adjacent_frames), we verify proximity.
    
    ball_shot_frames = []
    
    if clean_candidates:
        clean_candidates.sort()
        current_group = [clean_candidates[0]]
        
        # Iterate and group
        for i in range(1, len(clean_candidates)):
            frame = clean_candidates[i]
            prev_frame = current_group[-1]
            
            # If frames are close (within 24 frames / 1 sec), they belong to the same "Shot Event"
            if frame - prev_frame <= 24:
                current_group.append(frame)
            else:
                # Group finished -> Pick the Best Frame in this group
                best_frame = min(current_group, key=lambda x: get_proximity_score(ball_detections, player_detections, x))                
                ball_shot_frames.append(best_frame)
                
                # Start new group
                current_group = [frame]
        
        # Process the final group
        if current_group:
            best_frame = min(current_group, key=lambda x: get_proximity_score(ball_detections, player_detections, x))
            ball_shot_frames.append(best_frame)

    print(f"Refined Shots: {len(ball_shot_frames)} (Filtered noise by Proximity)")
    
    # FREE MEMORY: We are done with TrackNet. Unload it.
    print("Unloading Ball Tracker model to free VRAM...")
    del ball_tracker.model
    torch.cuda.empty_cache()

    # --- 4. COURT DETECTION (Use ENHANCED frames) ---
    # Lines are often faint, so contrast enhancement helps here too
    court_model_path = "/kaggle/input/cv-project/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    court_infer_interval = constants.COURT_INFER_INTERVAL
    print(f"Detecting court lines every {court_infer_interval} frames...")
    
    court_keypoints = []
    last_keypoints = None

    for i, frame in enumerate(enhanced_frames):
        if i % court_infer_interval == 0:
            last_keypoints = court_line_detector.predict(frame)
        
        # Safety check: if first frame fails, handle it (though unlikely)
        if last_keypoints is None:
            # Fallback to zeros or handle error if needed
            last_keypoints = [0] * 28 
            
        court_keypoints.append(last_keypoints)

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints[0], 
        player_detections, 
        player_detection_court_margin=player_detection_court_margin,
        last_known_positions=last_known_positions 
    )

    try:
        export_data = {}
        for i, frame_detections in enumerate(player_detections):
            frame_boxes = []
            for track_id, data in frame_detections.items():
                # data['bbox'] is [x1, y1, x2, y2]
                frame_boxes.append({
                    "track_id": track_id,
                    "bbox": data['bbox']
                })
            export_data[i] = frame_boxes

        # Save JSON alongside the output video path
        json_output_path = "kaggle/working/detections"
        os.makedirs(json_output_path, exist_ok=True)
        
        filename_with_ext = os.path.basename(original_video_name)
        file_root = filename_with_ext[0]  

        json_output_path = os.path.join(json_output_path, f"{file_root}_detections.json")
        
        with open(json_output_path, 'w') as f:
            json.dump(export_data, f, cls=NumpyEncoder, indent=4)
        print(f"   📄 Saved detection log to {json_output_path}")
    except Exception as e:
        print(f"   ⚠️ Could not save detection JSON: {e}")

    pose_estimator = YOLO('/kaggle/input/cv-project/yolo26x-pose.pt')

    print("Running Pose Estimation on detected players (BATCHED)...")
    
    # 1. Collect all crops from the entire video first
    all_crops = []
    crop_metadata = [] # Stores (frame_idx, track_id, crop_x1, crop_y1) to map back later
    
    for frame_idx, frame_dict in enumerate(player_detections):
        frame_img = enhanced_frames[frame_idx]
        img_h, img_w, _ = frame_img.shape
        
        for track_id, data in frame_dict.items():
            bbox = data['bbox']
            padding = constants.BOUNDING_BOX_PADDING
            
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_w, x2 + padding)
            y2 = min(img_h, y2 + padding)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                frame_dict[track_id]['keypoints'] = []
                continue

            # Crop and Store
            player_crop = frame_img[y1:y2, x1:x2]
            all_crops.append(player_crop)
            crop_metadata.append((frame_idx, track_id, x1, y1))

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

    # MiniCourt
    mini_court = MiniCourt(raw_frames[0]) 

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # --- FIX: CONVERT POINTS TO BOXES ---
    # MiniCourt and Draw functions expect Bounding Boxes [x1, y1, x2, y2],
    # but our new Tracker returns Center Points (x, y).
    # We create a fake 20x20 box around the center.
    ball_detections_boxes = []
    for pos in ball_detections:
        if pos is None or pos[0] is None or np.isnan(pos[0]):
            # Use an empty box inside a dictionary
            ball_detections_boxes.append({1: [0, 0, 0, 0]}) 
        else:
            x, y = pos
            pad = 10 
            # WRAP IN DICT: {1: [x1, y1, x2, y2]}
            ball_detections_boxes.append({1: [x-pad, y-pad, x+pad, y+pad]})
    
    ball_detections = ball_detections_boxes
    # ------------------------------------

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
                                                                            player_detections, 
                                                                            ball_detections,
                                                                            court_keypoints
                                                                            )
    
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
        'shot_type': None
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)):
        start_frame = ball_shot_frames[ball_shot_ind]
        
        if ball_shot_ind == len(ball_shot_frames) - 1:
            # Case: This is the LAST detected shot. 
            # We don't have a "next hit" to calculate speed, so we set defaults.
            speed_of_ball_shot = 0 
            ball_shot_time_in_seconds = 1 # Dummy value to avoid division by zero
            
            # We assume the "end" is just a bit later to capture the swing
            end_frame = min(len(enhanced_frames) - 1, start_frame + 20)
            
            # We cannot measure ball distance since we don't know where it lands
            distance_covered_by_ball_meters = 0
        else:
            # Case: Normal shot (Start -> End)
            end_frame = ball_shot_frames[ball_shot_ind+1]
            ball_shot_time_in_seconds = (end_frame-start_frame)/24

            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()
                                                                            ) 
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]

        # Safety check: if no players detected in this frame
        if len(player_positions) == 0:
            continue

        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # Map to 1 or 2
        mapped_shooter_id = player_id_map.get(player_shot_ball, 1)

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame

        # 1. Define the Window
        # The GCN needs a sequence (e.g., 40 frames). Center it on the shot frame.
        window_size = HDGCN_window_size
        half_window = window_size // 2
        start_window = max(0, start_frame - half_window)
        end_window = min(len(enhanced_frames), start_frame + half_window)

        # 2. Extract Keypoints Sequence (CORRECTED)
        sequence_data = []
        for f in range(start_window, end_window):
            # Check if frame exists and player is detected
            if f < len(player_detections) and player_shot_ball in player_detections[f]:
                # We need BOTH bbox and keypoints for normalization
                data_point = {
                    'bbox': player_detections[f][player_shot_ball]['bbox'],
                    'keypoints': player_detections[f][player_shot_ball]['keypoints']
                }
                sequence_data.append(data_point)
            else:
                # Handle missing frames (pad with dummy data)
                # We use a dummy bbox [0,0,1,1] to avoid division by zero errors
                sequence_data.append({'bbox': [0,0,1,1], 'keypoints': [[0,0,0]] * 17})

        # 3. Normalize using the class instance
        # You need to initialize 'extractor = PoseExtractor()' before the loop (see Fix #4)
        normalized_input = extractor.process_sequence(sequence_data)
        
        # Convert to tensor (N, C, T, V)
        inp_tensor = torch.from_numpy(normalized_input).unsqueeze(0).float()
        inp_tensor = inp_tensor.permute(0, 3, 1, 2) # (1, 3, 40, 17)

        # 4. Predict
        with torch.no_grad():
            output = action_model(inp_tensor)
            
            # --- FIX 1: BASELINE VOLLEY HALLUCINATION ---
            # Logic: If player is far from the net (> 4m), they cannot be hitting a volley.
            player_mc_pos = player_mini_court_detections[start_frame][player_shot_ball]
            
            # Calculate Net Y position (Midpoint of the court drawing)
            net_y = (mini_court.court_start_y + mini_court.court_end_y) / 2
            
            # Distance from Net
            dist_from_net_pixels = abs(player_mc_pos[1] - net_y)
            dist_from_net_meters = convert_pixel_distance_to_meters(
                dist_from_net_pixels, 
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )
            
            if dist_from_net_meters > 4.0: # If > 4 meters from net
                 for idx, class_name in enumerate(constants.THETIS_CLASSES):
                     if "volley" in class_name:
                         output[0][idx] = -float('inf')

            # --- FIX 2: SERVICE & SMASH CONFUSION (HEIGHT CHECK) ---
            # Logic: Serves/Smashes happen ABOVE the head. If ball is below nose, ban them.
            
            # Get Nose Y (Keypoint 0)
            shooter_kpts = player_detections[start_frame][player_shot_ball].get('keypoints', [])
            if shooter_kpts and len(shooter_kpts) > 0:
                nose_y = shooter_kpts[0][1]
                
                # Get Ball Y (Center of box)
                ball_box = ball_detections[start_frame][1]
                ball_y = (ball_box[1] + ball_box[3]) / 2
                
                # Image Coordinates: Y increases downwards.
                # So if Ball Y > Nose Y, the ball is BELOW the nose.
                if ball_y > nose_y:
                    for idx, class_name in enumerate(constants.THETIS_CLASSES):
                        if "service" in class_name or "smash" in class_name:
                            output[0][idx] = -float('inf')

            # Ban Serve after FRAME_LIMIT frames
            if start_frame > constants.FRAME_LIMIT_FOR_SERVES:
                # If a class name contains "service", kill its probability.
                for idx, class_name in enumerate(constants.THETIS_CLASSES):
                    if "service" in class_name:
                        # Set logit to negative infinity so argmax never picks it
                        output[0][idx] = -float('inf')

            prediction_idx = torch.argmax(output, dim=1).item()
            shot_name = constants.THETIS_CLASSES[prediction_idx]

        # SAVE TO LOG
        # Add frame_offset to start_frame so it matches the original full video
        true_frame_index = start_frame + frame_offset
        model_predictions_log.append({
            "frame": true_frame_index,
            "shot": shot_name,
            "player": mapped_shooter_id
        })

        current_player_stats['shot_type'] = shot_name

        current_player_stats['shot_player_id'] = mapped_shooter_id

        print(f"Frame {start_frame}: | Prediction: {shot_name} | Player: {player_shot_ball} (Mapped: {mapped_shooter_id})")
        
        # D. Opponent Speed (CRITICAL FIX FOR KEYERROR 2)
        # We find valid opponents present in the CURRENT frame
        current_players = list(player_mini_court_detections[start_frame].keys())
        opponents = [pid for pid in current_players if pid != player_shot_ball]
        
        speed_of_opponent = 0
        if len(opponents) > 0:
            opponent_player_id = opponents[0] # Use the actual detected opponent ID
            
            # Only measure speed if opponent is also in end frame
            if opponent_player_id in player_mini_court_detections[end_frame]:
                dist_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                               player_mini_court_detections[end_frame][opponent_player_id])
                dist_meters = convert_pixel_distance_to_meters(dist_pixels,
                                                               constants.DOUBLE_LINE_WIDTH,
                                                               mini_court.get_width_of_mini_court()) 
                speed_of_opponent = dist_meters/ball_shot_time_in_seconds * 3.6
                
                # Update stats for mapped opponent ID
                mapped_opponent_id = player_id_map.get(opponent_player_id, 2 if mapped_shooter_id == 1 else 1)
                current_player_stats[f'player_{mapped_opponent_id}_total_player_speed'] += speed_of_opponent
                current_player_stats[f'player_{mapped_opponent_id}_last_player_speed'] = speed_of_opponent

        # Update Shooter Stats
        current_player_stats[f'player_{mapped_shooter_id}_number_of_shots'] += 1
        current_player_stats[f'player_{mapped_shooter_id}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{mapped_shooter_id}_last_shot_speed'] = speed_of_ball_shot

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(enhanced_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(raw_frames, player_detections)
    output_video_frames = draw_skeletons(output_video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    # Pass the full list 'court_keypoints'
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    #output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    # Define Minimap dimensions (Fixed width from MiniCourt class)
    minimap_width = mini_court.drawing_rectangle_width
    minimap_start_x = mini_court.start_x
    minimap_end_y = mini_court.end_y
    
    for i, frame in enumerate(output_video_frames):
        current_stats = player_stats_data_df.iloc[i]
        shot_type = current_stats['shot_type']
        
        # Check if we have a valid shot type
        if shot_type is not None and str(shot_type) != 'nan':
            
            # 1. Format the text: "Forehand Flat" instead of "forehand_flat"
            shot_name = str(shot_type).replace('_', ' ').title()
            
            # 2. Add Player ID: "Forehand Flat (P1)"
            # Use .get() to avoid errors if the column is missing
            shot_player_id = current_stats.get('shot_player_id')
            
            if shot_player_id is not None and str(shot_player_id) != 'nan':
                # Convert to int (handles 1.0 -> 1)
                p_id = int(float(shot_player_id))
                text = f"{shot_name} (P{p_id})"
            else:
                text = f"{shot_name}"
            
            # 3. Font Settings (Normalized Size)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            
            # Use a FIXED, larger font scale by default for consistency
            font_scale = 0.65 
            
            # 4. Safety Check: Only shrink if it physically doesn't fit the box
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            while text_width > minimap_width and font_scale > 0.4:
                font_scale -= 0.05
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 5. Center Text horizontally relative to Minimap
            text_x = int(minimap_start_x + (minimap_width - text_width) / 2)
            text_y = int(minimap_end_y + 30 + text_height) # Padding below minimap

            # 6. Draw (Black Outline + White Text)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use the 'output_path' argument passed to the function
    save_video(output_video_frames, output_path)
    
    # TIMER
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Processing speed: {len(output_video_frames)/elapsed_time:.2f} FPS")

    # --- CALCULATE FINAL POSITIONS FOR NEXT CLIP ---
    final_positions = {}
    if len(player_detections) > 0:
        # Look at the last frame with detections
        for i in range(len(player_detections) - 1, -1, -1):
            if player_detections[i]:
                last_frame_detections = player_detections[i]
                for track_id, data in last_frame_detections.items():
                    # Only save positions for mapped players
                    if track_id in player_id_map:
                        p_num = player_id_map[track_id]
                        bbox = data['bbox']
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        final_positions[p_num] = [cx, cy]
                break

    return model_predictions_log, final_positions

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process a video file from a specific path using a specific window size and YOLO verbosity.")
    
    # Add the path argument
    parser.add_argument("--path", type=str, default = "/kaggle/input/tennis-rally-videos/input_video.mp4", help="The full path to the video file", required=True)

    parser.add_argument("--output-path", type=str, default = "/kaggle/working/output_videos", help="The full path to the output", required=False)

    parser.add_argument("--window-size", type=int, default = 40, help="The HDGCN shot recognition window size", required=True)

    parser.add_argument("--yolo-verbosity", action='store_true', help="Enable YOLO verbose logging")

    parser.add_argument("--player-detection-court-margin", type=int, default = 300, help="Court margin for detecting players and excluding line judges (in pixels)", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # 1. LOAD GLOBAL GROUND TRUTH ONCE
    ground_truth_path = args.path.rsplit(".", 1)[0] + ".json"
    gold_standard_data = []
    if ground_truth_path and os.path.exists(ground_truth_path):
        print(f"Loading Global Ground Truth from: {ground_truth_path}")
        with open(ground_truth_path, 'r') as f:
            gold_standard_data = json.load(f)

    # 2. SPLIT VIDEO
    temp_clip_dir = constants.TEMP_CLIP_DIR
    processed_clip_dir = constants.PROCESSED_CLIP_DIR
    
    # Call your new function here
    clip_paths = split_video_into_clips(args.path, temp_clip_dir, constants.CLIP_DURATION)
    
    # Ensure output directory exists
    os.makedirs(processed_clip_dir, exist_ok=True)

    # 3. PROCESS CLIPS SEQUENTIALLY
    all_model_predictions = []
    last_clip_positions = None

    # Store the paths of the *processed* clips to merge them later
    processed_files_list = []

    # ONLY sort if we actually split the video (files match the pattern clip_X_Y)
    # If it's the original file (len=1), we skip sorting to avoid the ValueError.
    if len(clip_paths) > 1:
        try:
            clip_paths.sort(key=lambda x: int(x.split('_')[-2]))
        except (ValueError, IndexError):
            print("Warning: Could not sort clips by index. Processing in default order.")

    for clip_path in clip_paths:
        filename = os.path.basename(clip_path)
        print(f"\n--- Processing Clip: {filename} ---")

        # Extract Frame Offset safely
        # Expecting format: clip_{chunk_idx}_{start_frame}.mp4
        try:
            # Check if this looks like one of our generated clips
            if filename.startswith("clip_") and "_" in filename:
                start_frame_str = filename.split('_')[-1].split('.')[0]
                frame_offset = int(start_frame_str)
            else:
                # It's the original video (shorter than 30s)
                frame_offset = 0
        except ValueError:
            print("Error parsing frame offset. Defaulting to 0.")
            frame_offset = 0

        # Define output path for this specific clip
        output_clip_path = os.path.join(processed_clip_dir, f"processed_{filename}")

        processed_files_list.append(output_clip_path)

        original_file = os.path.basename(args.path)
        original_video_name = os.path.splitext(original_file)

        # Process the clip
        clip_preds_list, last_clip_positions = process_single_clip(
            input_video=clip_path,
            output_path=output_clip_path,
            HDGCN_window_size=args.window_size,
            yolo_verbosity=args.yolo_verbosity,
            player_detection_court_margin=args.player_detection_court_margin,
            original_video_name=original_video_name,
            frame_offset=frame_offset,
            last_known_positions=last_clip_positions
        )
        
        # Aggregate stats
        all_model_predictions.extend(clip_preds_list)

    # 4. FINAL REPORT
    print("\n--- All Clips Processed. Generating Final Report ---")
    
    # Compare the aggregated predictions against the global ground truth
    print_validation_report(all_model_predictions, gold_standard_data)
    
    # MERGE LOGIC
    print("\n--- Merging Processed Clips ---")
    final_output_dir = args.output_path
    
    # Ensure the folder exists
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    original_filename = os.path.basename(args.path)
    final_output_path = os.path.join(final_output_dir, original_filename)

    merge_clips(processed_files_list, final_output_path)
    
    # Cleanup temp folder
    shutil.rmtree(temp_clip_dir)
    shutil.rmtree(processed_clip_dir)