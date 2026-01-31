import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get original FPS and set Target FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 24
    
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

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()