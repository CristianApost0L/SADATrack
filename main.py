import constants
import cv2
import pandas as pd
import numpy as np
import torch
import os
import argparse
import time
import json
import shutil
from ultralytics import YOLO 
from trackers import PlayerTracker, BallTracker, BounceDetector 
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from action_recognition.model import HDGCN_Tennis
from action_recognition.extractor import PoseExtractor
from action_recognition.shot_predictor import predict_shot
from action_recognition.pose_estimator import estimate_poses
from utils import (read_video, 
                   save_video,
                   draw_skeletons,
                   enhance_video_contrast,
                   print_validation_report,
                   split_video_into_clips,
                   merge_clips,
                   draw_shot_name_marker_frame_number,
                   export_json
                   )

def process_single_clip(input_video, 
                        output_path, 
                        HDGCN_window_size, 
                        yolo_verbosity, 
                        player_detection_court_margin, 
                        original_video_name, 
                        p1_handedness, 
                        p2_handedness, 
                        frame_offset=0, 
                        last_known_positions=None):
    start_time = time.time()

    model_predictions_log = []

    # Read Video
    input_video_path = input_video

    # Initialize Action Classifier
    extractor = PoseExtractor()

    action_model = HDGCN_Tennis(num_classes=12, in_channels=3)
    action_model.load_state_dict(torch.load(constants.ACTION_MODEL_PATH))
    action_model.eval()
      
    # --- DUAL STREAM SETUP ---
    # Stream A: Raw Frames (Clean, low noise) -> BEST FOR BALL DETECTION
    raw_frames = read_video(input_video_path)
    
    # Stream B: Enhanced Frames (High contrast) -> BEST FOR PLAYER/COURT DETECTION
    print("Preprocessing video for lighting/shadows...")
    enhanced_frames = enhance_video_contrast(raw_frames)


    # --- 2. DETECT PLAYERS (Use ENHANCED frames) ---
    player_tracker = PlayerTracker(model_path=constants.PLAYER_TRACKER_PATH)
    print("Detecting Players on Enhanced Video...")
    player_detections = player_tracker.detect_frames(enhanced_frames,
                                                     yolo_verbosity=yolo_verbosity
                                                     )
    
    print("Unloading Player Tracker model to free VRAM...")
    del player_tracker.model 
    torch.cuda.empty_cache()

    # --- 3. DETECT BALL (Use RAW frames) ---
    # This ignores the noisy/grainy enhanced frames and looks at the clean original
    ball_tracker = BallTracker(model_path=constants.BALL_TRACKER_PATH) # Amin model
    bounce_detector = BounceDetector(model_path=constants.BOUNCE_TRACKER_PATH) # Amin model

    print("Detecting Ball on Raw Video...")
    ball_detections, ball_shot_frames, detected_bounces = ball_tracker.find_ball_shot_frames(
        raw_frames, 
        player_detections, 
        bounce_detector
    )
    print("Unloading Ball Tracker model to free VRAM...")
    del ball_tracker.model
    torch.cuda.empty_cache()

    # --- 4. COURT DETECTION (Use ENHANCED frames) ---
    # Lines are often faint, so contrast enhancement helps here too
    court_line_detector = CourtLineDetector(constants.COURT_DETECTOR_PATH)
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

    # --- FILTER PLAYERS  ---
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints[0], 
        player_detections, 
        player_detection_court_margin=player_detection_court_margin,
        last_known_positions=last_known_positions 
    )

    # --- POSE ESTIMATION ---
    print("Running Pose Estimation on detected players (BATCHED)...")
    pose_estimator = YOLO(constants.YOLO_POSE_PATH)

    player_id_map, player_detections = estimate_poses(pose_estimator, 
                                                      player_detections, 
                                                      enhanced_frames, 
                                                      last_known_positions)

    export_json(player_id_map, player_detections, original_video_name)

    # MiniCourt
    mini_court = MiniCourt(raw_frames[0]) 

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
    
    bounce_events = {}
    for frame_idx in detected_bounces:
        # 1. Get Video Position (Center of the ball box)
        # Recall ball_detections is now a list of dicts: [{1: [x1,y1,x2,y2]}, ...]
        if frame_idx < len(ball_detections) and 1 in ball_detections[frame_idx]:
            bbox = ball_detections[frame_idx][1]
            video_pos = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        else:
            continue # Skip if no ball detected in bounce frame

        # 2. Get MiniMap Position
        if frame_idx < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[frame_idx]:
            mini_pos = ball_mini_court_detections[frame_idx][1]
            mini_pos = (int(mini_pos[0]), int(mini_pos[1]))
        else:
            continue

        bounce_events[frame_idx] = {'video': video_pos, 'mini': mini_pos}

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
    
    # --- PREDICT EACH SHOT ---
    for ball_shot_ind in range(len(ball_shot_frames)):
        # Get the latest stats to pass in
        last_stats = player_stats_data[-1]
        
        prediction_result = predict_shot(extractor,
                 action_model,
                 HDGCN_window_size,
                 ball_shot_frames, 
                 ball_shot_ind, 
                 enhanced_frames, 
                 mini_court,
                 player_detections,
                 p1_handedness,
                 p2_handedness,
                 frame_offset,
                 ball_detections,
                 player_mini_court_detections,
                 player_id_map,
                 ball_mini_court_detections,
                 last_stats)
        
        # Handle result
        if prediction_result is not None:
            shot_name, new_stats, log_entry = prediction_result
            
            # Update lists here (Main Loop Control)
            player_stats_data.append(new_stats)
            model_predictions_log.append(log_entry)


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

    draw_shot_name_marker_frame_number(output_video_frames, bounce_events, mini_court, player_stats_data_df)

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
    parser.add_argument("--p1-handedness", type=str, choices=['right', 'left'], default='right', help="Player 1 (Bottom) handedness")
    parser.add_argument("--p2-handedness", type=str, choices=['right', 'left'], default='right', help="Player 2 (Top) handedness")

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
            p1_handedness=args.p1_handedness,
            p2_handedness=args.p2_handedness,
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