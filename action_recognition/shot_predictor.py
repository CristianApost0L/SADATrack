import sys
sys.path.append('../')

import constants
import torch
from copy import deepcopy
from utils import (
    measure_distance,
    convert_pixel_distance_to_meters
)

def predict_shot(extractor,
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
                 player_stats_data,
                 ball_mini_court_detections,
                 model_predictions_log):
    '''
    Predicts the shot given data
    '''
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
        return

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

        # RIGHTY/LEFTY MASK
        # 1. Determine current shooter's handedness
        shooter_hand = p1_handedness if mapped_shooter_id == 1 else p2_handedness

        # 2. Get positions (Center X)
        p_bbox = player_detections[start_frame][player_shot_ball]['bbox']
        p_center_x = (p_bbox[0] + p_bbox[2]) / 2

        b_box = ball_detections[start_frame][1]
        b_center_x = (b_box[0] + b_box[2]) / 2

        # 3. Determine if ball is on the "Forehand Side" geometrically
        # Note: P1 (Bottom) faces UP (North). P2 (Top) faces DOWN (South).
        is_forehand_side = False

        if mapped_shooter_id == 1: # Bottom Player (Faces Away/Up)
            if shooter_hand == 'right':
                # Righty facing up: Ball on Right (Screen X > Player X) is Forehand
                if b_center_x > p_center_x: is_forehand_side = True
            else: 
                # Lefty facing up: Ball on Left (Screen X < Player X) is Forehand
                if b_center_x < p_center_x: is_forehand_side = True

        else: # Top Player (Faces Camera/Down)
            if shooter_hand == 'right':
                # Righty facing down: Ball on Screen LEFT is their Right side (Forehand)
                if b_center_x < p_center_x: is_forehand_side = True
            else:
                # Lefty facing down: Ball on Screen RIGHT is their Left side (Forehand)
                if b_center_x > p_center_x: is_forehand_side = True

        # 4. Apply Mask
        if is_forehand_side:
            # If geometry says Forehand, ban Backhand classes
            for idx, class_name in enumerate(constants.THETIS_CLASSES):
                if "backhand" in class_name:
                        output[0][idx] = -float('inf')
        else:
            # If geometry says Backhand, ban Forehand classes
            for idx, class_name in enumerate(constants.THETIS_CLASSES):
                if "forehand" in class_name:
                        output[0][idx] = -float('inf')
        # -----------------------------

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