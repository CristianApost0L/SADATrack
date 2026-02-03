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
                   split_video_into_clips
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
from ultralytics import YOLO 

def process_video_sequence(input_video_path, frame_offset, HDGCN_window_size, yolo_verbosity, player_detection_court_margin):
    """
    Processes a single video file (a full video or a clip).
    Returns the prediction log for this sequence and the number of frames processed.
    """
    print(f"Processing sequence: {input_video_path} (Offset: {frame_offset})")
    
    model_predictions_log = []

    # Read Video
    # --- DUAL STREAM SETUP ---
    # Stream A: Raw Frames (Clean, low noise) -> BEST FOR BALL DETECTION
    raw_frames = read_video(input_video_path)
    
    if len(raw_frames) == 0:
        return [], 0

    # Stream B: Enhanced Frames (High contrast) -> BEST FOR PLAYER/COURT DETECTION
    print("Preprocessing video for lighting/shadows...")
    enhanced_frames = enhance_video_contrast(raw_frames)

    # Initialize Action Classifier
    extractor = PoseExtractor()
    action_model = HDGCN_Tennis(num_classes=12, in_channels=3)
    action_model.load_state_dict(torch.load('/kaggle/input/cv-project/new_Swing_classifier.pth'))
    action_model.eval()

    # Initialize Trackers
    player_tracker = PlayerTracker(model_path='/kaggle/input/cv-project/yolo26x.pt')
    ball_tracker = BallTracker(model_path='/kaggle/input/cv-project/ball_model_best.pt') 
    bounce_detector = BounceDetector(model_path='/kaggle/input/cv-project/ctb_regr_bounce.cbm')

    # --- 2. DETECT PLAYERS (Use ENHANCED frames) ---
    print("Detecting Players on Enhanced Video...")
    player_detections = player_tracker.detect_frames(enhanced_frames, yolo_verbosity=yolo_verbosity)
    
    # FREE MEMORY
    print("Unloading Player Tracker model to free VRAM...")
    del player_tracker.model 
    torch.cuda.empty_cache()

    # --- 3. DETECT BALL (Use RAW frames) ---
    print("Detecting Ball on Raw Video...")
    ball_detections = ball_tracker.detect_frames(raw_frames)
    
    # Interpolate ball
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Get candidates (Hits + Bounces)
    candidate_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    candidate_shot_frames = filter_adjacent_frames(candidate_shot_frames, min_distance=24)

    # Detect Bounces
    detected_bounces = bounce_detector.predict(ball_detections) 

    # Filter: Keep a candidate ONLY if it is NOT a bounce
    clean_candidates = []
    for frame in candidate_shot_frames:
        is_bounce = False
        for b_frame in detected_bounces:
            if abs(frame - b_frame) <= 3: 
                is_bounce = True
                break
        if not is_bounce:
            clean_candidates.append(frame)

    # SMART FILTERING: Group frames and pick the one CLOSEST to a player
    ball_shot_frames = []
    if clean_candidates:
        clean_candidates.sort()
        current_group = [clean_candidates[0]]
        for i in range(1, len(clean_candidates)):
            frame = clean_candidates[i]
            prev_frame = current_group[-1]
            if frame - prev_frame <= 24:
                current_group.append(frame)
            else:
                best_frame = min(current_group, key=lambda x: get_proximity_score(ball_detections, player_detections, x))                
                ball_shot_frames.append(best_frame)
                current_group = [frame]
        if current_group:
            best_frame = min(current_group, key=lambda x: get_proximity_score(ball_detections, player_detections, x))
            ball_shot_frames.append(best_frame)

    print(f"Refined Shots: {len(ball_shot_frames)}")
    
    # FREE MEMORY
    print("Unloading Ball Tracker model to free VRAM...")
    del ball_tracker.model
    torch.cuda.empty_cache()

    # --- 4. COURT DETECTION ---
    court_model_path = "/kaggle/input/cv-project/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    court_infer_interval = constants.COURT_INFER_INTERVAL
    print(f"Detecting court lines every {court_infer_interval} frames...")
    
    court_keypoints = []
    last_keypoints = None

    for i, frame in enumerate(enhanced_frames):
        if i % court_infer_interval == 0:
            last_keypoints = court_line_detector.predict(frame)
        if last_keypoints is None:
            last_keypoints = [0] * 28 
        court_keypoints.append(last_keypoints)

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints[0], player_detections, player_detection_court_margin = player_detection_court_margin)

    pose_estimator = YOLO('/kaggle/input/cv-project/yolo26x-pose.pt')
    print("Running Pose Estimation on detected players (BATCHED)...")
    
    all_crops = []
    crop_metadata = [] 
    
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
            
            if x2 <= x1 or y2 <= y1:
                frame_dict[track_id]['keypoints'] = []
                continue

            player_crop = frame_img[y1:y2, x1:x2]
            all_crops.append(player_crop)
            crop_metadata.append((frame_idx, track_id, x1, y1))

    BATCH_SIZE = constants.BATCH_SIZE
    all_pose_results = []
    
    for i in range(0, len(all_crops), BATCH_SIZE):
        batch_crops = all_crops[i : i + BATCH_SIZE]
        batch_results = pose_estimator(batch_crops, verbose=False, stream=False)
        all_pose_results.extend(batch_results)

    for i, result in enumerate(all_pose_results):
        frame_idx, track_id, crop_x1, crop_y1 = crop_metadata[i]
        found_keypoints = False
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kpts = result.keypoints.data[0].cpu().numpy()
            kpts[:, 0] += crop_x1
            kpts[:, 1] += crop_y1
            player_detections[frame_idx][track_id]['keypoints'] = kpts.tolist()
            found_keypoints = True
        
        if not found_keypoints:
            player_detections[frame_idx][track_id]['keypoints'] = []

    print("Smoothing skeleton keypoints...")
    player_detections = smooth_keypoints(player_detections)

    # --- DYNAMIC ID MAPPING ---
    id_occupancy = {}
    for frame_dict in player_detections:
        for track_id in frame_dict.keys():
            id_occupancy[track_id] = id_occupancy.get(track_id, 0) + 1
            
    valid_ids = sorted(id_occupancy, key=id_occupancy.get, reverse=True)[:2]
    id_avg_height = {}
    
    for pid in valid_ids:
        heights = []
        for frame_dict in player_detections:
            if pid in frame_dict:
                bbox = frame_dict[pid]['bbox']
                h = bbox[3] - bbox[1]
                heights.append(h)
        if heights:
            id_avg_height[pid] = sum(heights) / len(heights)
        else:
            id_avg_height[pid] = 0

    sorted_ids = sorted(valid_ids, key=lambda x: id_avg_height.get(x, 0), reverse=True)
    
    player_id_map = {} 
    if len(sorted_ids) >= 1: 
        player_id_map[sorted_ids[0]] = 1
    if len(sorted_ids) >= 2: 
        player_id_map[sorted_ids[1]] = 2
    
    all_detected_ids = set(id_occupancy.keys())
    for pid in all_detected_ids:
        if pid not in player_id_map:
            player_id_map[pid] = 1

    # MiniCourt
    mini_court = MiniCourt(raw_frames[0]) 

    ball_detections_boxes = []
    for pos in ball_detections:
        if pos is None or pos[0] is None or np.isnan(pos[0]):
            ball_detections_boxes.append({1: [0, 0, 0, 0]}) 
        else:
            x, y = pos
            pad = 10 
            ball_detections_boxes.append({1: [x-pad, y-pad, x+pad, y+pad]})
    
    ball_detections = ball_detections_boxes

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
            speed_of_ball_shot = 0 
            ball_shot_time_in_seconds = 1
            end_frame = min(len(enhanced_frames) - 1, start_frame + 20)
        else:
            end_frame = ball_shot_frames[ball_shot_ind+1]
            ball_shot_time_in_seconds = (end_frame-start_frame)/24
            distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court()
                                                                            ) 
            speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        player_positions = player_mini_court_detections[start_frame]

        if len(player_positions) == 0:
            continue

        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))
        mapped_shooter_id = player_id_map.get(player_shot_ball, 1)

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame

        # Action Recognition
        window_size = HDGCN_window_size
        half_window = window_size // 2
        start_window = max(0, start_frame - half_window)
        end_window = min(len(enhanced_frames), start_frame + half_window)

        sequence_data = []
        for f in range(start_window, end_window):
            if f < len(player_detections) and player_shot_ball in player_detections[f]:
                data_point = {
                    'bbox': player_detections[f][player_shot_ball]['bbox'],
                    'keypoints': player_detections[f][player_shot_ball]['keypoints']
                }
                sequence_data.append(data_point)
            else:
                sequence_data.append({'bbox': [0,0,1,1], 'keypoints': [[0,0,0]] * 17})

        normalized_input = extractor.process_sequence(sequence_data)
        inp_tensor = torch.from_numpy(normalized_input).unsqueeze(0).float()
        inp_tensor = inp_tensor.permute(0, 3, 1, 2) 

        with torch.no_grad():
            output = action_model(inp_tensor)
            
            player_mc_pos = player_mini_court_detections[start_frame][player_shot_ball]
            net_y = (mini_court.court_start_y + mini_court.court_end_y) / 2
            dist_from_net_pixels = abs(player_mc_pos[1] - net_y)
            dist_from_net_meters = convert_pixel_distance_to_meters(
                dist_from_net_pixels, 
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )
            
            if dist_from_net_meters > 4.0: 
                 for idx, class_name in enumerate(constants.THETIS_CLASSES):
                     if "volley" in class_name:
                         output[0][idx] = -float('inf')

            shooter_kpts = player_detections[start_frame][player_shot_ball].get('keypoints', [])
            if shooter_kpts and len(shooter_kpts) > 0:
                nose_y = shooter_kpts[0][1]
                ball_box = ball_detections[start_frame][1]
                ball_y = (ball_box[1] + ball_box[3]) / 2
                if ball_y > nose_y:
                    for idx, class_name in enumerate(constants.THETIS_CLASSES):
                        if "service" in class_name or "smash" in class_name:
                            output[0][idx] = -float('inf')

            # IMPORTANT: For global frame consistency, we use frame_offset
            # The 'start_frame' variable is local to this clip.
            # However, the logic for FRAME_LIMIT_FOR_SERVES might depend on the start of the match.
            # Assuming 'serves' are only valid at the very start of a rally, this might be tricky with clips.
            # But we stick to the local clip frame for this specific logic unless we track rally state.
            if (start_frame + frame_offset) > constants.FRAME_LIMIT_FOR_SERVES:
                for idx, class_name in enumerate(constants.THETIS_CLASSES):
                    if "service" in class_name:
                        output[0][idx] = -float('inf')

            prediction_idx = torch.argmax(output, dim=1).item()
            shot_name = constants.THETIS_CLASSES[prediction_idx]

        # SAVE TO LOG with GLOBAL OFFSET
        model_predictions_log.append({
            "frame": start_frame + frame_offset,
            "shot": shot_name,
            "player": mapped_shooter_id
        })

        current_player_stats['shot_type'] = shot_name
        current_player_stats['shot_player_id'] = mapped_shooter_id
        
        current_players = list(player_mini_court_detections[start_frame].keys())
        opponents = [pid for pid in current_players if pid != player_shot_ball]
        
        speed_of_opponent = 0
        if len(opponents) > 0:
            opponent_player_id = opponents[0] 
            if opponent_player_id in player_mini_court_detections[end_frame]:
                dist_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                               player_mini_court_detections[end_frame][opponent_player_id])
                dist_meters = convert_pixel_distance_to_meters(dist_pixels,
                                                               constants.DOUBLE_LINE_WIDTH,
                                                               mini_court.get_width_of_mini_court()) 
                speed_of_opponent = dist_meters/ball_shot_time_in_seconds * 3.6
                mapped_opponent_id = player_id_map.get(opponent_player_id, 2 if mapped_shooter_id == 1 else 1)
                current_player_stats[f'player_{mapped_opponent_id}_total_player_speed'] += speed_of_opponent
                current_player_stats[f'player_{mapped_opponent_id}_last_player_speed'] = speed_of_opponent

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
    output_video_frames = player_tracker.draw_bboxes(raw_frames, player_detections)
    output_video_frames = draw_skeletons(output_video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Stats on frames
    # output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df) # Optional/Commented in original

    minimap_width = mini_court.drawing_rectangle_width
    minimap_start_x = mini_court.start_x
    minimap_end_y = mini_court.end_y
    
    for i, frame in enumerate(output_video_frames):
        current_stats = player_stats_data_df.iloc[i]
        shot_type = current_stats['shot_type']
        
        if shot_type is not None and str(shot_type) != 'nan':
            shot_name = str(shot_type).replace('_', ' ').title()
            shot_player_id = current_stats.get('shot_player_id')
            
            if shot_player_id is not None and str(shot_player_id) != 'nan':
                p_id = int(float(shot_player_id))
                text = f"{shot_name} (P{p_id})"
            else:
                text = f"{shot_name}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            font_scale = 0.65 
            
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            while text_width > minimap_width and font_scale > 0.4:
                font_scale -= 0.05
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = int(minimap_start_x + (minimap_width - text_width) / 2)
            text_y = int(minimap_end_y + 30 + text_height)

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    ## Draw frame number on top left corner (Include Offset for display?)
    # For debugging, we can show local or global. Let's show Global frame num.
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i + frame_offset}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")

    # Generate output name based on input name
    base_name = os.path.basename(input_video_path)
    save_path = f"output_videos/output_{base_name}"
    # Change extension to .avi for compatibility with save_video function which uses MJPG
    save_path = os.path.splitext(save_path)[0] + ".avi"

    save_video(output_video_frames, save_path)
    
    return model_predictions_log, len(raw_frames)

def main(input_video, HDGCN_window_size, yolo_verbosity, player_detection_court_margin):
    start_time = time.time()

    # LOAD GROUND TRUTH FROM JSON (Global)
    ground_truth_path = input_video.rsplit(".", 1)[0] + ".json"
    gold_standard_data = []
    if ground_truth_path and os.path.exists(ground_truth_path):
        print(f"Loading Ground Truth labels from: {ground_truth_path}")
        with open(ground_truth_path, 'r') as f:
            gold_standard_data = json.load(f)
    elif ground_truth_path:
        print(f"Warning: Ground Truth file not found at {ground_truth_path}")

    # Check Video Duration
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error reading video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    global_predictions_log = []
    
    if duration <= 30:
        print(f"Video duration is {duration:.2f}s. Processing as single file.")
        logs, _ = process_video_sequence(input_video, 0, HDGCN_window_size, yolo_verbosity, player_detection_court_margin)
        global_predictions_log.extend(logs)
    else:
        print(f"Video duration is {duration:.2f}s (>30s). Splitting into clips...")
        clips = split_video_into_clips(input_video, clip_duration=30, output_dir="temp_clips")
        
        current_frame_offset = 0
        
        # Process sequentially
        for clip_path in clips:
            logs, frames_processed = process_video_sequence(clip_path, current_frame_offset, HDGCN_window_size, yolo_verbosity, player_detection_court_margin)
            global_predictions_log.extend(logs)
            
            # Update offset for next clip
            current_frame_offset += frames_processed
            
            # Clean up temp clip to save space
            os.remove(clip_path) 

    # TIMER
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

    # FINAL VALIDATION REPORT
    print_validation_report(global_predictions_log, gold_standard_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file from a specific path using a specific window size and YOLO verbosity.")
    parser.add_argument("--path", type=str, default = "input_videos/input_video.mp4", help="The full path to the video file", required=True)
    parser.add_argument("--window-size", type=int, default = 40, help="The HDGCN shot recognition window size", required=True)
    parser.add_argument("--yolo-verbosity", type=bool, default = False, help="YOLO log verbosity", required=True)
    parser.add_argument("--player-detection-court-margin", type=int, default = 300, help="Court margin for detecting players and excluding line judges (in pixels)", required=True)
    args = parser.parse_args()

    main(args.path, args.window_size, args.yolo_verbosity, args.player_detection_court_margin)