import os
import cv2
import shutil

def print_validation_report(model_predictions_log, gold_standard_data, frame_tolerance=10):
    """
    Compares prediction log against ground truth with fuzzy matching for shot types.
    Levels:
    - EXACT:   forehand_flat == forehand_flat
    - PARTIAL: forehand_flat ~= forehand_openstands (Same Side)
    - WRONG:   forehand_flat != backhand_slice (Wrong Side)
    """
    if not gold_standard_data:
        print("\nNo Ground Truth data provided. Skipping validation report.")
        return

    print("\n" + "="*60)
    print(f"{'FINAL ACCURACY REPORT':^60}")
    print("="*60)
    
    exact_shots = 0
    partial_shots = 0
    correct_players = 0
    total_labels = len(gold_standard_data)

    for gt in gold_standard_data:
        # 1. Find ALL candidates within tolerance
        candidates = []
        for pred in model_predictions_log:
            if abs(pred['frame'] - gt['frame']) <= frame_tolerance:
                candidates.append(pred)

        # 2. Select the BEST candidate based on a Score
        match = None
        if candidates:
            def get_score(cand):
                score = 0
                
                # A. Player Match (Highest Priority)
                if cand['player'] == gt['player']:
                    score += 1000
                
                # B. Shot Match (High Priority)
                gt_shot = gt['shot'].lower()
                pred_shot = cand['shot'].lower()
                
                if gt_shot == pred_shot:
                    score += 500
                else:
                    # Partial Match Check
                    is_forehand = "forehand" in gt_shot and "forehand" in pred_shot
                    is_backhand = "backhand" in gt_shot and "backhand" in pred_shot
                    is_serve    = ("serve" in gt_shot or "service" in gt_shot) and \
                                  ("serve" in pred_shot or "service" in pred_shot)
                    if is_forehand or is_backhand or is_serve:
                        score += 300
                
                # C. Time Proximity (Tie-breaker)
                # We subtract the distance, so closer frames have higher scores
                dist = abs(cand['frame'] - gt['frame'])
                score -= dist 
                
                return score

            # Pick the candidate with the max score
            match = max(candidates, key=get_score)

        print(f"Frame {gt['frame']:<4}: ", end="")
        
        if match:
            # Analyze Player ID
            player_ok = (match['player'] == gt['player'])
            if player_ok: correct_players += 1

            # Analyze Shot Type
            gt_shot = gt['shot'].lower()
            pred_shot = match['shot'].lower()
            
            shot_status = "WRONG"
            
            # Exact
            if gt_shot == pred_shot:
                shot_status = "EXACT"
                exact_shots += 1
            # Partial
            else:
                is_forehand = "forehand" in gt_shot and "forehand" in pred_shot
                is_backhand = "backhand" in gt_shot and "backhand" in pred_shot
                is_serve    = ("serve" in gt_shot or "service" in gt_shot) and \
                              ("serve" in pred_shot or "service" in pred_shot)
                
                if is_forehand or is_backhand or is_serve:
                    shot_status = "PARTIAL"
                    partial_shots += 1
            
            # Determine Icon
            if shot_status == "EXACT" and player_ok:
                icon = "✅ PERFECT"
            elif shot_status == "PARTIAL" and player_ok:
                icon = "⚠️ PARTIAL"
            elif shot_status == "WRONG" and player_ok:
                icon = "❌ WRONG SHOT"
            elif not player_ok:
                icon = "❌ WRONG PLAYER"
            
            print(f"{icon:<15} | Expected: {gt['shot']} (P{gt['player']})")
            print(f"{'':<12} Found:    {match['shot']} (P{match['player']}) @ Frame {match['frame']}")
            
        else:
            print(f"{'❌ MISSED':<15} | Expected: {gt['shot']} (P{gt['player']})")

    print("-" * 60)
    print(f"Player ID Accuracy:     {correct_players}/{total_labels} ({correct_players/total_labels*100:.1f}%)")
    print(f"Shot Exact Matches:     {exact_shots}/{total_labels} ({exact_shots/total_labels*100:.1f}%)")
    print(f"Shot Partial Matches:   {partial_shots}/{total_labels} (Total Useful: {(exact_shots+partial_shots)/total_labels*100:.1f}%)")
    print("="*60 + "\n")

def save_validation_clips(video_path, model_predictions_log, gold_standard_data, output_dir, frame_tolerance=10, clip_window=40):
    """
    Iterates through Ground Truth, matches predictions (using the same logic as the report),
    and saves a video clip for each event.
    """
    if not gold_standard_data:
        return

    if not os.path.exists(video_path):
        print(f"⚠️ Cannot find video file at {video_path}. Skipping clip generation.")
        return

    val_clips_dir = os.path.join(output_dir, "validation_clips")
    # delete folder (and everything inside)
    shutil.rmtree(val_clips_dir, ignore_errors=True)  # ignore if it doesn't exist
    # recreate folder
    os.makedirs(val_clips_dir, exist_ok=True)

    print(f"\n[INFO] Saving validation clips to: {val_clips_dir}")

    # Extract clean video name (e.g. "match_01.mp4" -> "match_01")
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 24
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for gt in gold_standard_data:
        # --- MATCHING LOGIC (Identical to print_validation_report) ---
        candidates = []
        for pred in model_predictions_log:
            if abs(pred['frame'] - gt['frame']) <= frame_tolerance:
                candidates.append(pred)

        match = None
        if candidates:
            def get_score(cand):
                score = 0
                if cand['player'] == gt['player']: score += 1000
                gt_shot = gt['shot'].lower()
                pred_shot = cand['shot'].lower()
                if gt_shot == pred_shot: score += 500
                else:
                    is_fh = "forehand" in gt_shot and "forehand" in pred_shot
                    is_bh = "backhand" in gt_shot and "backhand" in pred_shot
                    is_sv = ("serve" in gt_shot or "service" in gt_shot) and ("serve" in pred_shot or "service" in pred_shot)
                    if is_fh or is_bh or is_sv: score += 300
                dist = abs(cand['frame'] - gt['frame'])
                score -= dist 
                return score
            match = max(candidates, key=get_score)
        # -----------------------------------------------------------

        # Determine Filename components
        status_str = "MISSED"
        pred_shot_str = "None"
        center_frame = gt['frame'] # Default to GT if missed

        if match:
            center_frame = match['frame']
            player_ok = (match['player'] == gt['player'])
            gt_shot = gt['shot'].lower()
            pred_shot = match['shot'].lower()
            pred_shot_str = match['shot']

            if gt_shot == pred_shot and player_ok:
                status_str = "PERFECT"
            elif player_ok:
                is_fh = "forehand" in gt_shot and "forehand" in pred_shot
                is_bh = "backhand" in gt_shot and "backhand" in pred_shot
                is_sv = ("serve" in gt_shot or "service" in gt_shot) and ("serve" in pred_shot or "service" in pred_shot)
                if is_fh or is_bh or is_sv:
                    status_str = "PARTIAL"
                else:
                    status_str = "WRONG_SHOT"
            else:
                status_str = "WRONG_PLAYER"

        # Generate Clip
        clean_gt = gt['shot'].replace(" ", "_")
        clean_pred = pred_shot_str.replace(" ", "_")
        
        clip_name = f"{video_name}_Frame_{gt['frame']:04d}_{status_str}_Exp_{clean_gt}_Found_{clean_pred}.mp4"
        clip_path = os.path.join(val_clips_dir, clip_name)
        
        start_f = max(0, center_frame - clip_window)
        end_f = min(total_vid_frames - 1, center_frame + clip_window)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_clip = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        
        curr = start_f
        while curr <= end_f:
            ret, frame = cap.read()
            if not ret: break
            out_clip.write(frame)
            curr += 1
        out_clip.release()
        
    cap.release()
    print("Validation clips generation complete.")

def save_clean_validation_clips(original_video_path, model_predictions_log, gold_standard_data, output_dir, processed_fps=24, frame_tolerance=10, clip_window=40):
    """
    Saves clips from the ORIGINAL RAW video (Clean, No Overlays).
    Crucially, it RESAMPLES the raw video to match the processed frame rate (24fps).
    This ensures the 'clean' clip is exactly the same speed/duration as the 'processed' clip.
    """
    if not gold_standard_data: return

    if not os.path.exists(original_video_path):
        print(f"⚠️ Cannot find original video file at {original_video_path}. Skipping clean clip generation.")
        return

    val_clips_dir = os.path.join(output_dir, "validation_clips_clean")
    os.makedirs(val_clips_dir, exist_ok=True)
    print(f"\n[INFO] Saving CLEAN validation clips to: {val_clips_dir}")

    video_name = os.path.splitext(os.path.basename(original_video_path))[0]
    
    cap = cv2.VideoCapture(original_video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate Ratio to map Processed Frame Index -> Original Frame Index
    # e.g. If Processed=24fps, Orig=60fps, Ratio = 2.5
    # Processed Frame 100 corresponds to Original Frame 250
    fps_ratio = orig_fps / processed_fps

    for gt in gold_standard_data:
        # --- MATCHING LOGIC (COPIED EXACTLY) ---
        candidates = []
        for pred in model_predictions_log:
            if abs(pred['frame'] - gt['frame']) <= frame_tolerance:
                candidates.append(pred)

        match = None
        if candidates:
            def get_score(cand):
                score = 0
                if cand['player'] == gt['player']: score += 1000
                gt_shot = gt['shot'].lower()
                pred_shot = cand['shot'].lower()
                if gt_shot == pred_shot: score += 500
                else:
                    is_fh = "forehand" in gt_shot and "forehand" in pred_shot
                    is_bh = "backhand" in gt_shot and "backhand" in pred_shot
                    is_sv = ("serve" in gt_shot or "service" in gt_shot) and ("serve" in pred_shot or "service" in pred_shot)
                    if is_fh or is_bh or is_sv: score += 300
                dist = abs(cand['frame'] - gt['frame'])
                score -= dist 
                return score
            match = max(candidates, key=get_score)
        # ---------------------------------------

        status_str = "MISSED"
        pred_shot_str = "None"
        center_frame_processed = gt['frame']

        if match:
            center_frame_processed = match['frame']
            player_ok = (match['player'] == gt['player'])
            gt_shot = gt['shot'].lower()
            pred_shot = match['shot'].lower()
            pred_shot_str = match['shot']

            if gt_shot == pred_shot and player_ok: status_str = "PERFECT"
            elif player_ok:
                is_fh = "forehand" in gt_shot and "forehand" in pred_shot
                is_bh = "backhand" in gt_shot and "backhand" in pred_shot
                is_sv = ("serve" in gt_shot or "service" in gt_shot) and ("serve" in pred_shot or "service" in pred_shot)
                if is_fh or is_bh or is_sv: status_str = "PARTIAL"
                else: status_str = "WRONG_SHOT"
            else: status_str = "WRONG_PLAYER"

        clean_gt = gt['shot'].replace(" ", "_")
        clean_pred = pred_shot_str.replace(" ", "_")
        
        # Determine Window in Processed Time (Frames)
        start_f_proc = max(0, center_frame_processed - clip_window)
        end_f_proc = center_frame_processed + clip_window

        # Append _clean to filename
        clip_name = f"{video_name}_Frame_{gt['frame']:04d}_{status_str}_Exp_{clean_gt}_Found_{clean_pred}_clean.mp4"
        clip_path = os.path.join(val_clips_dir, clip_name)
        
        # Prepare Writer (at Processed FPS, e.g. 24)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_clip = cv2.VideoWriter(clip_path, fourcc, processed_fps, (width, height))
        
        # --- FRAME SAMPLING LOGIC ---
        # We iterate through the PROCESSED frame indices (start_f_proc -> end_f_proc)
        # For each, we calculate the corresponding ORIGINAL frame index, seek, and write.
        curr_proc = start_f_proc
        while curr_proc <= end_f_proc:
            # Map to original frame index
            orig_frame_idx = int(curr_proc * fps_ratio)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame_idx)
            ret, frame = cap.read()
            if not ret: 
                break
                
            out_clip.write(frame)
            curr_proc += 1
            
        out_clip.release()
        
    cap.release()
    print("Clean clips generation complete.")