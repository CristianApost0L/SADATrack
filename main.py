from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
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
from ultralytics import YOLO 

def draw_skeletons(video_frames, player_detections):
    output_frames = []
    # COCO Keypoint connections (standard skeleton structure)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
        (5, 11), (6, 12), (11, 12), # Torso
        (11, 13), (13, 15), (12, 14), (14, 16) # Legs
    ]
    
    for frame, player_dict in zip(video_frames, player_detections):
        for track_id, data in player_dict.items():
            # Get keypoints if they exist
            if 'keypoints' not in data:
                continue
                
            kpts = data['keypoints'] # List of [x, y, conf]
            
            # Skip if keypoints are missing or malformed
            if len(kpts) != 17:
                continue

            # Draw Lines (Limbs)
            for p1, p2 in connections:
                # Check confidence (index 2) -> if < 0.5, don't draw
                if kpts[p1][2] < 0.5 or kpts[p2][2] < 0.5:
                    continue
                
                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                
                # Draw limb in Green
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw Points (Joints)
            for i, kp in enumerate(kpts):
                if kp[2] < 0.5: continue
                x, y = int(kp[0]), int(kp[1])
                # Draw joint in Red
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                
        output_frames.append(frame)
    
    return output_frames

# The official 12 classes from the dataset
THETIS_CLASSES = [
    "backhand2hands", "backhand", "backhand_slice", "backhand_volley",
    "forehand_flat", "forehand_openstands", "forehand_slice", "forehand_volley",
    "flat_service", "kick_service", "slice_service", "smash"
]

def main(input_video):
    # Read Video
    input_video_path = input_video

    # Initialize Action Classifier
    extractor = PoseExtractor()

    action_model = HDGCN_Tennis(num_classes=12, in_channels=3)
    action_model.load_state_dict(torch.load('/kaggle/input/cv-project/new_Swing_classifier.pth'))
    action_model.eval()
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='/kaggle/input/cv-project/yolo26x.pt')
    ball_tracker = BallTracker(model_path='/kaggle/input/cv-project/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    # Court Line Detector model
    court_model_path = "/kaggle/input/cv-project/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    pose_estimator = YOLO('/kaggle/input/cv-project/yolo26x-pose.pt')

    print("Running Pose Estimation on detected players...")
    for frame_idx, frame_dict in enumerate(player_detections):
        frame_img = video_frames[frame_idx]
        img_h, img_w, _ = frame_img.shape
        
        for track_id, data in frame_dict.items():
            # Get the box detected by the Tracker
            bbox = data['bbox'] # [x1, y1, x2, y2]
            
            # Crop logic with boundary checks
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            # If box is invalid (zero width/height), skip
            if x2 <= x1 or y2 <= y1:
                frame_dict[track_id]['keypoints'] = []
                continue

            player_crop = frame_img[y1:y2, x1:x2]
            
            # Run Pose Estimation on the crop
            # verbose=False keeps the console clean
            results = pose_estimator(player_crop, verbose=False)[0]
            
            found_keypoints = False
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                # Extract the first skeleton found in the crop
                kpts = results.keypoints.data[0].cpu().numpy() # Shape (17, 3)
                
                # IMPORTANT: Transform crop coordinates back to full frame coordinates
                kpts[:, 0] += x1 # Add crop offset X
                kpts[:, 1] += y1 # Add crop offset Y
                
                # Update the detection dictionary with the real keypoints
                frame_dict[track_id]['keypoints'] = kpts.tolist()
                found_keypoints = True
            
            if not found_keypoints:
                # If pose model fails to find a person in the crop, set empty
                frame_dict[track_id]['keypoints'] = []

    # --- DYNAMIC ID MAPPING ---
    # Map the actual Track IDs (e.g. 5, 23) to "Player 1" and "Player 2"
    all_track_ids = set()
    for frame_dict in player_detections:
        all_track_ids.update(frame_dict.keys())
    sorted_ids = sorted(list(all_track_ids))
    
    player_id_map = {} # Maps TrackID -> 1 or 2
    if len(sorted_ids) >= 1: player_id_map[sorted_ids[0]] = 1
    if len(sorted_ids) >= 2: player_id_map[sorted_ids[1]] = 2
    # Map any extras to 1 to prevent crashes
    for pid in sorted_ids[2:]: player_id_map[pid] = 1

    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

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
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
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
        window_size = 40 
        half_window = window_size // 2
        start_window = max(0, start_frame - half_window)
        end_window = min(len(video_frames), start_frame + half_window)

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
            prediction_idx = torch.argmax(output, dim=1).item()
            shot_name = THETIS_CLASSES[prediction_idx]
            
        # --- DEBUG: CHECK IF INPUT IS EMPTY ---
        non_zero_frames = np.count_nonzero(normalized_input)
        print(f"Frame {start_frame}: Input Non-Zeros: {non_zero_frames} | Prediction: {shot_name} | Player: {player_shot_ball}")
        # --------------------------------------

        current_player_stats['shot_type'] = shot_name
        print(f"Frame {start_frame}: Input Non-Zeros: {non_zero_frames} | Prediction: {shot_name} | Player: {player_shot_ball} (Mapped: {mapped_shooter_id})")
        
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
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']



    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = draw_skeletons(output_video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    import os
    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process a video file from a specific path.")
    
    # Add the path argument
    parser.add_argument("--path", type=str, default = "input_videos/input_video.mp4", help="The full path to the video file", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Call main
    main(args.path)