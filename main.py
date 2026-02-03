import sys
import os
import cv2
import time
import json
import torch
import shutil
import glob
import argparse
import subprocess
import gc
import numpy as np
import pandas as pd
from copy import deepcopy
from ultralytics import YOLO 

# Appending path for local modules
sys.path.append('CV_project')

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
                   get_proximity_score
                   )
import constants
from trackers import PlayerTracker, BallTracker, BounceDetector 
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from action_recognition.model import HDGCN_Tennis
from action_recognition.extractor import PoseExtractor

# -----------------------------------------------------------
# HELPER: MEMORY CLEANUP
# -----------------------------------------------------------
def force_cleanup():
    """Forces aggressive garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------
# HELPER: VIDEO SPLITTING & STITCHING
# -----------------------------------------------------------
def split_video_ffmpeg(input_path, chunk_duration=30):
    """
    Splits video into physical chunks using FFmpeg to save RAM during read_video().
    Returns a list of temporary chunk file paths.
    """
    temp_dir = "temp_chunks"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Output pattern: temp_chunks/chunk_000.mp4
    output_pattern = os.path.join(temp_dir, "chunk_%03d.mp4")
    
    print(f"✂️ Splitting video into {chunk_duration}s chunks...")
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(chunk_duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]
    
    # Run silent ffmpeg
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Get list of created chunks
    chunks = sorted(glob.glob(os.path.join(temp_dir, "*.mp4")))
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks, temp_dir

def stitch_videos_ffmpeg(chunk_paths, output_path):
    """
    Stitches processed video chunks back together.
    """
    print("🧵 Stitching chunks together...")
    
    # Create inputs.txt for ffmpeg concat
    list_file = "inputs.txt"
    with open(list_file, "w") as f:
        for path in chunk_paths:
            # ffmpeg requires relative paths or safe absolute paths
            f.write(f"file '{path}'\n")
            
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy", "-y",
        output_path
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    # Cleanup
    if os.path.exists(list_file): os.remove(list_file)
    print(f"🎉 Final video saved to: {output_path}")

# -----------------------------------------------------------
# CORE: PROCESS SINGLE CHUNK
# -----------------------------------------------------------
def process_video_chunk(chunk_path, chunk_index, global_frame_offset, 
                        HDGCN_window_size, yolo_verbosity, player_detection_court_margin,
                        output_folder="temp_outputs"):
    
    print(f"\n🚀 PROCESSING CHUNK {chunk_index} (Global Offset: {global_frame_offset})...")
    
    # --- 1. DUAL STREAM SETUP ---
    # Stream A: Raw Frames (Clean, low noise) -> BEST FOR BALL DETECTION
    raw_frames = read_video(chunk_path)
    if not raw_frames:
        return [], 0, None
        
    # Stream B: Enhanced Frames (High contrast) -> BEST FOR PLAYER/COURT DETECTION
    print("Preprocessing video for lighting/shadows (CLAHE)...")
    enhanced_frames = enhance_video_contrast(raw_frames)
    
    chunk_predictions_log = []
    
    # --- 2. Initialize Models (Scoped to this function) ---
    # We load them here and destroy them at the end of the function to free RAM
    extractor = PoseExtractor()
    
    action_model = HDGCN_Tennis(num_classes=12, in_channels=3)
    action_model.load_state_dict(torch.load('/kaggle/input/cv-project/new_Swing_classifier.pth'))
    action_model.eval()

    player_tracker = PlayerTracker(model_path='/kaggle/input/cv-project/yolo26x.pt')
    ball_tracker = BallTracker(model_path='/kaggle/input/cv-project/ball_model_best.pt')
    bounce_detector = BounceDetector(model_path='/kaggle/input/cv-project/ctb_regr_bounce.cbm')
    
    # --- 3. DETECT PLAYERS (Use ENHANCED frames) ---
    print("Detecting Players on Enhanced Video...")
    player_detections = player_tracker.detect_frames(enhanced_frames, yolo_verbosity=yolo_verbosity)
    
    # FREE MEMORY: YOLO
    print("Unloading Player Tracker model to free VRAM...")
    del player_tracker.model 
    force_cleanup()

    # --- 4. DETECT BALL (Use RAW frames) ---
    print("Detecting Ball on Raw Video...")
    ball_detections = ball_tracker.detect_frames(raw_frames)
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # A. Get Candidates
    candidate_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    candidate_shot_frames = filter_adjacent_frames(candidate_shot_frames, min_distance=24)

    # B. Detect Bounces
    detected_bounces = bounce_detector.predict(ball_detections) 

    # C. Filter Bounces
    clean_candidates = []
    for frame in candidate_shot_frames:
        is_bounce = False
        for b_frame in detected_bounces:
            if abs(frame - b_frame) <= 3: 
                is_bounce = True; break
        if not is_bounce: clean_candidates.append(frame)

    # D. Smart Filtering (Proximity)
    ball_shot_frames = []
    if clean_candidates:
        clean_candidates.sort()
        current_group = [clean_candidates[0]]
        for i in range(1, len(clean_candidates)):
            frame = clean_candidates[i]
            if frame - current_group[-1] <= 24:
                current_group.append(frame)
            else:
                best = min(current_group, key=lambda x: get_proximity_score(ball_detections, player_detections, x))
                ball_shot_frames.append(best)
                current_group = [frame]
        if current_group:
            best = min(current_group, key=lambda x: get_proximity_score(ball_detections, player_detections, x))
            ball_shot_frames.append(best)

    print(f"Refined Shots: {len(ball_shot_frames)} (Filtered noise by Proximity)")
    
    # FREE MEMORY: Ball Tracker
    print("Unloading Ball Tracker model to free VRAM...")
    del ball_tracker.model
    force_cleanup()

    # --- 5. COURT DETECTION (Use ENHANCED frames) ---
    court_model_path = "/kaggle/input/cv-project/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    print(f"Detecting court lines...")
    court_keypoints = []
    last_keypoints = None
    for i, frame in enumerate(enhanced_frames):
        if i % constants.COURT_INFER_INTERVAL == 0:
            last_keypoints = court_line_detector.predict(frame)
        if last_keypoints is None: last_keypoints = [0]*28
        court_keypoints.append(last_keypoints)

    # Filter Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints[0], player_detections, player_detection_court_margin)

    # --- 6. POSE ESTIMATION (Batched - Uses ENHANCED frames) ---
    print("Running Pose Estimation on detected players (BATCHED)...")
    pose_estimator = YOLO('/kaggle/input/cv-project/yolo26x-pose.pt')
    
    all_crops = []
    crop_metadata = []
    for f_idx, f_dict in enumerate(player_detections):
        img_h, img_w = enhanced_frames[f_idx].shape[:2]
        for t_id, data in f_dict.items():
            bbox = data['bbox']
            pad = constants.BOUNDING_BOX_PADDING
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(img_w, x2+pad), min(img_h, y2+pad)
            
            if x2 > x1 and y2 > y1:
                all_crops.append(enhanced_frames[f_idx][y1:y2, x1:x2])
                crop_metadata.append((f_idx, t_id, x1, y1))

    BATCH_SIZE = constants.BATCH_SIZE
    all_pose_results = []
    for i in range(0, len(all_crops), BATCH_SIZE):
        batch = all_crops[i:i+BATCH_SIZE]
        results = pose_estimator(batch, verbose=False, stream=False)
        all_pose_results.extend(results)
        # Cleanup batch memory
        del batch, results
        force_cleanup()

    # Map back results
    for i, result in enumerate(all_pose_results):
        f_idx, t_id, cx, cy = crop_metadata[i]
        found = False
        if result.keypoints and len(result.keypoints.data) > 0:
            kpts = result.keypoints.data[0].cpu().numpy()
            kpts[:, 0] += cx; kpts[:, 1] += cy
            player_detections[f_idx][t_id]['keypoints'] = kpts.tolist()
            found = True
        if not found: player_detections[f_idx][t_id]['keypoints'] = []

    del pose_estimator
    force_cleanup()

    print("Smoothing skeleton keypoints...")
    player_detections = smooth_keypoints(player_detections)

    # --- 7. DYNAMIC ID MAPPING ---
    id_occupancy = {}
    for fd in player_detections:
        for tid in fd: id_occupancy[tid] = id_occupancy.get(tid, 0) + 1
    valid_ids = sorted(id_occupancy, key=id_occupancy.get, reverse=True)[:2]
    
    id_avg_height = {}
    for pid in valid_ids:
        heights = [fd[pid]['bbox'][3]-fd[pid]['bbox'][1] for fd in player_detections if pid in fd]
        id_avg_height[pid] = np.mean(heights) if heights else 0

    sorted_ids = sorted(valid_ids, key=lambda x: id_avg_height.get(x, 0), reverse=True)
    player_id_map = {pid: 1 for pid in id_occupancy} 
    if len(sorted_ids) >= 1: player_id_map[sorted_ids[0]] = 1
    if len(sorted_ids) >= 2: player_id_map[sorted_ids[1]] = 2

    # --- 8. LOGIC & ACTION RECOGNITION ---
    mini_court = MiniCourt(raw_frames[0])
    
    # Fix Ball Boxes (Convert points to boxes)
    ball_detections_boxes = []
    for pos in ball_detections:
        if pos and pos[0] is not None:
             ball_detections_boxes.append({1: [pos[0]-10, pos[1]-10, pos[0]+10, pos[1]+10]})
        else:
             ball_detections_boxes.append({1: [0,0,0,0]})
    ball_detections = ball_detections_boxes

    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
                                                                            player_detections, 
                                                                            ball_detections,
                                                                            court_keypoints)

    player_stats_data = [{
        'frame_num':0, 'player_1_number_of_shots':0, 'player_1_total_shot_speed':0, 'player_1_last_shot_speed':0, 'player_1_total_player_speed':0, 'player_1_last_player_speed':0,
        'player_2_number_of_shots':0, 'player_2_total_shot_speed':0, 'player_2_last_shot_speed':0, 'player_2_total_player_speed':0, 'player_2_last_player_speed':0,
        'shot_type': None
    }]

    for ball_shot_ind in range(len(ball_shot_frames)):
        start_frame = ball_shot_frames[ball_shot_ind]
        
        # Calculate Speed
        if ball_shot_ind == len(ball_shot_frames) - 1:
            speed_of_ball_shot = 0; ball_shot_time_in_seconds = 1
            end_frame = min(len(enhanced_frames) - 1, start_frame + 20)
        else:
            end_frame = ball_shot_frames[ball_shot_ind+1]
            ball_shot_time_in_seconds = (end_frame-start_frame)/24
            dist_pixels = measure_distance(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
            dist_meters = convert_pixel_distance_to_meters(dist_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court()) 
            speed_of_ball_shot = dist_meters/ball_shot_time_in_seconds * 3.6

        # Determine Shooter
        p_pos = player_mini_court_detections[start_frame]
        if not p_pos: continue
        player_shot_ball = min(p_pos.keys(), key=lambda pid: measure_distance(p_pos[pid], ball_mini_court_detections[start_frame][1]))
        mapped_shooter_id = player_id_map.get(player_shot_ball, 1)

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame

        # Prepare Input for GCN
        win_size = HDGCN_window_size
        s_win = max(0, start_frame - win_size//2)
        e_win = min(len(enhanced_frames), start_frame + win_size//2)
        
        seq = []
        for f in range(s_win, e_win):
            if f < len(player_detections) and player_shot_ball in player_detections[f]:
                seq.append({'bbox': player_detections[f][player_shot_ball]['bbox'], 'keypoints': player_detections[f][player_shot_ball]['keypoints']})
            else:
                seq.append({'bbox': [0,0,1,1], 'keypoints': [[0,0,0]]*17})
        
        inp = torch.from_numpy(extractor.process_sequence(seq)).unsqueeze(0).float().permute(0,3,1,2)
        
        with torch.no_grad():
            output = action_model(inp)
            
            # --- FILTER 1: BASELINE VOLLEY ---
            shooter_y = player_mini_court_detections[start_frame][player_shot_ball][1]
            net_y = (mini_court.court_start_y + mini_court.court_end_y) / 2
            dist_net = convert_pixel_distance_to_meters(abs(shooter_y - net_y), constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
            if dist_net > 4.0:
                 for idx, cls in enumerate(constants.THETIS_CLASSES):
                     if "volley" in cls: output[0][idx] = -float('inf')

            # --- FILTER 2: SERVICE HEIGHT ---
            kpts = player_detections[start_frame][player_shot_ball].get('keypoints', [])
            if kpts:
                nose_y = kpts[0][1]
                ball_y = (ball_detections[start_frame][1][1] + ball_detections[start_frame][1][3]) / 2
                if ball_y > nose_y:
                    for idx, cls in enumerate(constants.THETIS_CLASSES):
                        if "service" in cls or "smash" in cls: output[0][idx] = -float('inf')

            # --- FILTER 3: FRAME LIMIT ---
            if start_frame > constants.FRAME_LIMIT_FOR_SERVES:
                for idx, cls in enumerate(constants.THETIS_CLASSES):
                    if "service" in cls: output[0][idx] = -float('inf')

            pred_idx = torch.argmax(output, dim=1).item()
            shot_name = constants.THETIS_CLASSES[pred_idx]

        # Log GLOBAL Frame
        model_predictions_log.append({
            "frame": global_frame_offset + start_frame,
            "shot": shot_name,
            "player": mapped_shooter_id
        })

        current_player_stats['shot_type'] = shot_name
        current_player_stats['shot_player_id'] = mapped_shooter_id
        
        # Calculate Opponent Speed (Original logic maintained)
        current_players = list(player_mini_court_detections[start_frame].keys())
        opponents = [pid for pid in current_players if pid != player_shot_ball]
        speed_of_opponent = 0
        if len(opponents) > 0:
            opponent_id = opponents[0]
            if opponent_id in player_mini_court_detections[end_frame]:
                d_pix = measure_distance(player_mini_court_detections[start_frame][opponent_id], player_mini_court_detections[end_frame][opponent_id])
                d_met = convert_pixel_distance_to_meters(d_pix, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court()) 
                speed_of_opponent = d_met/ball_shot_time_in_seconds * 3.6
                
                map_opp_id = player_id_map.get(opponent_id, 2 if mapped_shooter_id == 1 else 1)
                current_player_stats[f'player_{map_opp_id}_total_player_speed'] += speed_of_opponent
                current_player_stats[f'player_{map_opp_id}_last_player_speed'] = speed_of_opponent

        # Update Stats
        current_player_stats[f'player_{mapped_shooter_id}_number_of_shots'] += 1
        current_player_stats[f'player_{mapped_shooter_id}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{mapped_shooter_id}_last_shot_speed'] = speed_of_ball_shot
        player_stats_data.append(current_player_stats)

    # --- 9. DRAW OUTPUT ---
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(enhanced_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left').ffill()

    # Fill averages
    for p in [1, 2]:
        n_shots = player_stats_data_df[f'player_{p}_number_of_shots'].replace(0, 1)
        player_stats_data_df[f'player_{p}_average_shot_speed'] = player_stats_data_df[f'player_{p}_total_shot_speed'] / n_shots
        n_opp_shots = player_stats_data_df[f'player_{3-p}_number_of_shots'].replace(0, 1)
        player_stats_data_df[f'player_{p}_average_player_speed'] = player_stats_data_df[f'player_{p}_total_player_speed'] / n_opp_shots

    # Note: draw_bboxes uses raw_frames (the base video)
    output_frames = player_tracker.draw_bboxes(raw_frames, player_detections)
    output_frames = draw_skeletons(output_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections)
    output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, color=(0,255,255))
    
    # Draw Stats Text
    mm_width = mini_court.drawing_rectangle_width
    mm_start_x = mini_court.start_x
    mm_end_y = mini_court.end_y
    
    for i, frame in enumerate(output_frames):
        row = player_stats_data_df.iloc[i]
        shot_type = row['shot_type']
        
        if shot_type and str(shot_type) != 'nan':
            shot_name = str(shot_type).replace('_', ' ').title()
            pid = row.get('shot_player_id')
            p_text = f" (P{int(float(pid))})" if pid and str(pid)!='nan' else ""
            text = f"{shot_name}{p_text}"
            
            font_scale = 0.65
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            while tw > mm_width and font_scale > 0.4:
                font_scale -= 0.05
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            
            tx = int(mm_start_x + (mm_width - tw) / 2)
            ty = int(mm_end_y + 30 + th)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4)
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Draw Frame Number
        cv2.putText(frame, f"Frame: {global_frame_offset + i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save Chunk Video
    os.makedirs(output_folder, exist_ok=True)
    chunk_out_path = os.path.join(output_folder, f"out_chunk_{chunk_index:03d}.avi")
    save_video(output_frames, chunk_out_path)
    
    # cleanup local memory
    del output_frames, raw_frames, enhanced_frames
    force_cleanup()
    
    return chunk_predictions_log, chunk_out_path

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
def main(input_video, HDGCN_window_size, yolo_verbosity, player_detection_court_margin):
    start_time = time.time()
    
    # Check Video Duration
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"🎥 Input Video: {duration:.2f} seconds ({total_frames} frames)")
    
    global_predictions_log = []
    chunk_output_files = []
    
    # DECISION: SPLIT OR NOT?
    if duration > 30:
        print("⚠️ Video > 30s. Engaging Chunking Mode to save RAM.")
        chunks, temp_dir = split_video_ffmpeg(input_video, chunk_duration=30)
        
        global_offset = 0
        for i, chunk_path in enumerate(chunks):
            # Process
            log, out_vid = process_video_chunk(
                chunk_path, i, global_offset,
                HDGCN_window_size, yolo_verbosity, player_detection_court_margin
            )
            
            global_predictions_log.extend(log)
            chunk_output_files.append(out_vid)
            
            # Update offset based on actual frames in this chunk
            cap_chunk = cv2.VideoCapture(chunk_path)
            chunk_frames = int(cap_chunk.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_chunk.release()
            global_offset += chunk_frames
            
            # Clean temp input chunk to save disk space
            os.remove(chunk_path)
            
        # Stitch
        if not os.path.exists("output_videos"): os.makedirs("output_videos")
        final_out = "output_videos/output_video.avi"
        stitch_videos_ffmpeg(chunk_output_files, final_out)
        
        # Cleanup Outputs
        shutil.rmtree(temp_dir)
        shutil.rmtree("temp_outputs")
        
    else:
        print("✅ Video < 30s. Processing as single chunk.")
        log, out_vid = process_video_chunk(
            input_video, 0, 0,
            HDGCN_window_size, yolo_verbosity, player_detection_court_margin,
            output_folder="output_videos"
        )
        global_predictions_log = log
        # Rename output to standard name
        if os.path.exists("output_videos/output_video.avi"):
            os.remove("output_videos/output_video.avi")
        os.rename(out_vid, "output_videos/output_video.avi")

    # Load Ground Truth for Report
    ground_truth_path = input_video.rsplit(".", 1)[0] + ".json"
    gold_standard_data = []
    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r') as f:
            gold_standard_data = json.load(f)

    # FINAL VALIDATION REPORT
    print_validation_report(global_predictions_log, gold_standard_data)
    
    elapsed = time.time() - start_time
    print(f"\n🏁 Total Time: {elapsed:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--yolo-verbosity", type=bool, default=False)
    parser.add_argument("--player-detection-court-margin", type=int, default=300)
    args = parser.parse_args()
    
    main(args.path, args.window_size, args.yolo_verbosity, args.player_detection_court_margin)