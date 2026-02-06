import cv2
import constants
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get original FPS and set Target FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = constants.TARGET_FPS
    
    # Calculate the ratio (Source / Target)
    # e.g. 60 / 24 = 2.5 (Skip frames)
    # e.g. 12 / 24 = 0.5 (Duplicate frames)
    ratio = original_fps / target_fps
    
    frames = []
    current_source_frame = 0
    target_frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # If the video is already close to 24fps (e.g. 23.97, 24, 25), accept it as is
        if abs(original_fps - target_fps) < 2:
            frames.append(frame)
        else:
            # Resampling Logic:
            # We determine which source frame corresponds to the current target frame.
            # int(target_frame_index * ratio) gives the index of the source frame we need.
            
            # While the current source frame is the one we need for the output...
            while int(target_frame_index * ratio) == current_source_frame:
                frames.append(frame)
                target_frame_index += 1
                
        current_source_frame += 1
    
    cap.release()
    
    # Optional: Print info if resampling occurred
    if abs(original_fps - target_fps) >= 2:
        print(f"[INFO] Video resampled from {original_fps:.2f} FPS to {target_fps} FPS.")
        print(f"       Original frames: {current_source_frame}, New frames: {len(frames)}")

    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        print("No frames to save.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # Ensure dimensions match the frames (Height, Width) vs (Width, Height) logic
    height, width, _ = output_video_frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def merge_clips(clip_paths, output_path, fps=24):
    """
    Merges multiple video clips into a single video file.
    """
    if not clip_paths:
        print("No clips to merge.")
        return

    print(f"Merging {len(clip_paths)} clips into {output_path}...")

    # Read the first clip to get dimensions
    cap = cv2.VideoCapture(clip_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for clip in clip_paths:
        print(f"  - Appending {os.path.basename(clip)}")
        cap = cv2.VideoCapture(clip)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print("Merge complete.")

def draw_skeletons(video_frames, player_detections, ):
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

def enhance_video_contrast(frames):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    to a list of video frames to handle shadows and uneven lighting.
    """
    enhanced_frames = []
    
    # Create CLAHE object
    # clipLimit -> Threshold for contrast limiting (2.0 is standard, higher = more contrast but more noise)
    # tileGridSize -> Size of grid for histogram equalization (8x8 is standard)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print(f"Enhancing contrast for {len(frames)} frames...")

    for frame in frames:
        # 1. Convert BGR to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # 2. Split into channels (L = Lightness, A/B = Colors)
        l, a, b = cv2.split(lab)
        
        # 3. Apply CLAHE to the L-channel
        l_enhanced = clahe.apply(l)
        
        # 4. Merge back and convert to BGR
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        enhanced_frames.append(frame_enhanced)
        
    return enhanced_frames

def split_video_into_clips(video_path, output_dir, clip_duration=180):
    """
    Splits a video into clips of a specified duration with start-frame naming.
    
    Args:
        video_path (str): Path to the input video.
        clip_duration (int): Duration of each clip in seconds.
        output_dir (str): Directory to save the clips.
    
    Returns:
        list: A list of paths to the generated clip files.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return []

    # 1. Get FPS and Total Frame Count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. Calculate total duration in seconds
    if fps > 0:
        video_length_seconds = total_frames / fps
    else:
        print("Error: Could not determine FPS.")
        return []

    # 3. Check if video is shorter than the target clip duration
    if video_length_seconds < clip_duration:
        print(f"Video is {video_length_seconds:.2f}s (shorter than {clip_duration}s). Skipping split.")
        cap.release()
        # Return the original path since no splitting occurred
        return [video_path]

    frames_per_clip = int(fps * clip_duration)
    clip_paths = []
    chunk_idx = 1 
    curr_frame = 0

    print(f"Splitting video into {clip_duration}s clips...")

    while curr_frame < total_frames:
        start_frame = curr_frame
        clip_name = f"clip_{chunk_idx}_{start_frame}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        clip_paths.append(clip_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(clip_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        
        processed_frames = 0
        while processed_frames < frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            processed_frames += 1
            curr_frame += 1
            
        out.release()
        chunk_idx += 1
        
    cap.release()
    return clip_paths
  
def draw_shot_name_marker_frame_number(output_video_frames, bounce_events, mini_court, player_stats_data_df):
    '''
    Draws on the video: name of the shot, marker for the last two bounces and frame number
    '''
    current_bounce_markers = {'top': None, 'bottom': None}
    # Calculate Net Y on MiniMap to distinguish halves
    # MiniCourt.court_start_y is top, court_end_y is bottom. Net is average.
    minimap_net_y = (mini_court.court_start_y + mini_court.court_end_y) / 2
    # Define Minimap dimensions (Fixed width from MiniCourt class)
    minimap_width = mini_court.drawing_rectangle_width
    minimap_start_x = mini_court.start_x
    minimap_end_y = mini_court.end_y
    
    for i, frame in enumerate(output_video_frames):
        # 1. Check if a new bounce happened in this frame
        if i in bounce_events:
            evt = bounce_events[i]
            vid_pos = evt['video']
            mini_pos = evt['mini']
            
            # Determine Half: Compare Y coordinate on MiniMap against Net Y
            if mini_pos[1] < minimap_net_y:
                current_bounce_markers['top'] = (vid_pos, mini_pos)
            else:
                current_bounce_markers['bottom'] = (vid_pos, mini_pos)

        # 2. Draw Markers (White X)
        for half, markers in current_bounce_markers.items():
            if markers is None: continue
            
            vid_pt, mini_pt = markers
            
            # Draw X on Main Video
            idx_size = 10
            cv2.line(frame, (vid_pt[0]-idx_size, vid_pt[1]-idx_size), (vid_pt[0]+idx_size, vid_pt[1]+idx_size), (255,255,255), 2)
            cv2.line(frame, (vid_pt[0]+idx_size, vid_pt[1]-idx_size), (vid_pt[0]-idx_size, vid_pt[1]+idx_size), (255,255,255), 2)

            # Draw X on MiniMap
            # Note: MiniMap is already drawn on 'frame', so we just draw on top of it at mini_pt coordinates
            mini_idx_size = 5
            cv2.line(frame, (mini_pt[0]-mini_idx_size, mini_pt[1]-mini_idx_size), (mini_pt[0]+mini_idx_size, mini_pt[1]+mini_idx_size), (255,255,255), 2)
            cv2.line(frame, (mini_pt[0]+mini_idx_size, mini_pt[1]-mini_idx_size), (mini_pt[0]-mini_idx_size, mini_pt[1]+mini_idx_size), (255,255,255), 2)

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

def crop_players(player_detections, enhanced_frames):
    '''
    Crops players (their bounding box) out of videos for pose dection
    '''
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
    
    return all_crops, crop_metadata