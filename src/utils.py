from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import cv2
import numpy as np
import pandas as pd

def scene_detect(path_video):
    """
    Split video to disjoint fragments based on color histograms
    """
    video_manager = VideoManager([path_video])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    if scene_list == []:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    scenes = [[x[0].frame_num, x[1].frame_num]for x in scene_list]    
    return scenes

def read_video(path_video):
    """
    Read video frames from file
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    print(f"Reading video: {path_video}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    return frames, fps

class PreciseCoordinateExtractor:
    """
    Converts pixel data from this specific repo's CourtReference 
    into real-world meters relative to the net.
    """
    def __init__(self):
        # Constants derived from the 'CourtReference' class in the repo
        # Top Baseline Y: 561, Bottom Baseline Y: 2935 -> Length in px: 2374
        # Real Tennis Court Length: 23.77 meters
        # Scale: ~99.87 pixels per meter. We'll use 100 for simplicity.
        self.px_per_meter = 100.0 
        
        # Center of the court in the reference image
        # X Center = (Left Line 286 + Right Line 1379) / 2 = 832.5
        # Y Center = Net Line = 1748
        self.ref_net_y = 1748
        self.ref_center_x = 832.5

    def get_coordinates(self, bounces, ball_track, homography_matrices):
        data = []
        print("Extracting coordinates for detected bounces...")
        
        for frame_idx in sorted(list(bounces)):
            # 1. Get Ball Position
            ball_pos = ball_track[frame_idx]
            
            # Handle missing ball (look +/- 1 frame)
            if ball_pos[0] is None:
                if frame_idx > 0 and ball_track[frame_idx-1][0]: 
                    ball_pos = ball_track[frame_idx-1]
                elif frame_idx < len(ball_track)-1 and ball_track[frame_idx+1][0]:
                    ball_pos = ball_track[frame_idx+1]
                else:
                    continue # Skip if ball is lost
            
            # 2. Get Homography (Video -> Reference)
            # The repo's 'infer_model' returns the inverted matrix (Video -> Ref)
            inv_matrix = homography_matrices[frame_idx]
            
            if inv_matrix is not None:
                # 3. Project to Reference Image
                pt_original = np.array([ball_pos], dtype='float32').reshape(1, 1, 2)
                pt_ref = cv2.perspectiveTransform(pt_original, inv_matrix)[0][0]
                
                # 4. Convert to Meters (Relative to Net)
                # Y: Negative = Far Court, Positive = Near Court
                real_y = (pt_ref[1] - self.ref_net_y) / self.px_per_meter
                # X: Negative = Left, Positive = Right
                real_x = (pt_ref[0] - self.ref_center_x) / self.px_per_meter
                
                # 5. Determine Zones
                depth = "Service Box" if abs(real_y) < 6.40 else ("Deep" if abs(real_y) < 11.89 else "Out")
                side = "Deuce" if (real_x * real_y) < 0 else "Ad" # Standard tennis logic

                data.append({
                    'frame': frame_idx,
                    'x_meters': round(real_x, 2),
                    'y_meters': round(real_y, 2),
                    'depth': depth,
                    'side': side
                })
        
        return pd.DataFrame(data)

def evaluate_models(reference_bounces,
                       candidate_bounces,
                       tolerance_before=2,
                       tolerance_after=6,
                       beta=1.0):
    """
    Human-aligned evaluation for bounce detection.

    :param reference_bounces: Ground Truth (human annotated)
    :param candidate_bounces: Detector predictions
    :param tolerance_before: Allowed frames BEFORE GT (detector early)
    :param tolerance_after:  Allowed frames AFTER GT (detector late)
    :param beta: F-beta score (beta > 1 favors recall)
    """

    ref = np.array(sorted(reference_bounces))
    cand = np.array(sorted(candidate_bounces))

    tp = []
    fp = []
    fn = []
    errors = []

    ref_idx = 0
    matched_refs = set()

    for c in cand:
        matched = False

        while ref_idx < len(ref):
            delta = c - ref[ref_idx]

            # Candidate too early → try next candidate
            if delta < -tolerance_before:
                break

            # Valid match (human-perceptual window)
            if -tolerance_before <= delta <= tolerance_after:
                tp.append(c)
                errors.append(delta)
                matched_refs.add(ref_idx)
                ref_idx += 1
                matched = True
                break

            # Candidate too late → reference missed
            fn.append(ref[ref_idx])
            ref_idx += 1

        if not matched:
            fp.append(c)

    # Remaining unmatched references → FN
    for i in range(ref_idx, len(ref)):
        if i not in matched_refs:
            fn.append(ref[i])

    # Metrics
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0

    if precision + recall > 0:
        f_beta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
    else:
        f_beta = 0

    mean_error = np.mean(errors) if errors else 0
    mean_abs_error = np.mean(np.abs(errors)) if errors else 0

    return {
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1 Score": round(f_beta, 3),
        "Mean Bias (frames)": round(mean_error, 2),
        "Mean Abs Error": round(mean_abs_error, 2),
        "TP": len(tp),
        "FP": len(fp),
        "FN": len(fn)
    }

def create_heatmap_overlay_video(video_path, output_path, df_data, homography_matrices):
    from .court_reference import CourtReference
    
    # 1. Setup Video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 2. Setup Robust Minimap
    court_ref = CourtReference()
    
    # Create blank canvas
    base_h = court_ref.court_total_height
    base_w = court_ref.court_total_width
    court_bg = np.zeros((base_h, base_w, 3), dtype=np.uint8)
    
    # Drawing Settings for Overlay
    LINE_THICKNESS = 25 
    LINE_COLOR = (255, 255, 255) # Pure White
    
    # Helper to draw lines from the reference points
    def draw_line(img, pts):
        pt1 = (int(pts[0][0]), int(pts[0][1]))
        pt2 = (int(pts[1][0]), int(pts[1][1]))
        cv2.line(img, pt1, pt2, LINE_COLOR, LINE_THICKNESS)

    # Draw Standard Lines
    draw_line(court_bg, court_ref.baseline_top)
    draw_line(court_bg, court_ref.baseline_bottom)
    draw_line(court_bg, court_ref.net)
    draw_line(court_bg, court_ref.top_inner_line)
    draw_line(court_bg, court_ref.bottom_inner_line)
    draw_line(court_bg, court_ref.left_court_line)
    draw_line(court_bg, court_ref.right_court_line)
    draw_line(court_bg, court_ref.left_inner_line)
    draw_line(court_bg, court_ref.right_inner_line)
    draw_line(court_bg, court_ref.middle_line)
    
    # Draw Center Marks
    cv2.line(court_bg, (int(832.5), int(561)), (int(832.5), int(580)), LINE_COLOR, LINE_THICKNESS)
    cv2.line(court_bg, (int(832.5), int(2935)), (int(832.5), int(2910)), LINE_COLOR, LINE_THICKNESS)

    # Resize settings
    mini_h = int(height * 0.35) 
    scale = mini_h / base_h
    mini_w = int(base_w * scale)
    
    print(f"Generating Overlay Video with Enhanced Lines: {output_path}...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. PREPARE MINIMAP ---
        minimap = court_bg.copy()
        dots_layer = np.zeros_like(minimap)
        
        # Get bounces
        current_bounces = df_data[df_data['frame'] <= frame_idx]
        
        for _, bounce in current_bounces.iterrows():
            mx = (bounce['x_meters'] * 100) + 832.5
            my = (bounce['y_meters'] * 100) + 1748
            color = (0, 255, 0) if bounce['side'] == 'Deuce' else (0, 0, 255)
            cv2.circle(dots_layer, (int(mx), int(my)), 50, color, -1)

        # Blend dots onto court
        minimap = cv2.addWeighted(minimap, 1.0, dots_layer, 0.6, 0)
        
        # --- B. RESIZE & OVERLAY ---
        minimap_small = cv2.resize(minimap, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
        
        margin = 20
        x1 = width - mini_w - margin
        y1 = margin
        x2 = x1 + mini_w
        y2 = y1 + mini_h
        
        # Safe Check for bounds
        if y2 < height and x2 < width:
            roi = frame[y1:y2, x1:x2]
            blended = cv2.addWeighted(roi, 0.3, minimap_small, 0.7, 0)
            frame[y1:y2, x1:x2] = blended
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Done.")

class MatchStatsEngine:
    def __init__(self, swing_model):
        self.swing_model = swing_model
        self.court_width = 10.97
        self.service_line_depth = 6.40
        self.baseline_depth = 11.89

    def process_rally(self, ball_track, bounces, skeletons, homography_matrices):
        events = []
        
        # 1. Identify "Hits" (Racket Impacts)
        hit_frames = self.detect_hits(ball_track, skeletons)
        
        # 2. Iterate through events chronologically
        for i in range(len(hit_frames) - 1):
            frame_hit = hit_frames[i]
            frame_next_hit = hit_frames[i+1]
            
            # Find the bounce between these two hits
            bounce_frame = next((b for b in bounces if frame_hit < b < frame_next_hit), None)
            
            if bounce_frame:
                # --- A. CLASSIFY SWING ---
                clip = self.extract_clip(skeletons, frame_hit)
                shot_type = self.swing_model.predict(clip)
                
                # --- B. CALCULATE COORDINATES ---
                matrix = homography_matrices[bounce_frame]
                ball_pix = ball_track[bounce_frame]
                
                if matrix is not None and ball_pix[0] is not None:
                     bounce_pos = self.project_point(ball_pix, matrix)
                     
                     depth = self.get_depth_label(bounce_pos)
                     direction = self.get_direction(ball_track[frame_hit], ball_track[bounce_frame], matrix)
                     
                     events.append({
                         'frame': frame_hit,
                         'shot': shot_type,
                         'depth': depth,
                         'direction': direction,
                         'x': bounce_pos[0],
                         'y': bounce_pos[1]
                     })
        
        return pd.DataFrame(events)

    def get_depth_label(self, pos):
        y = abs(pos[1])
        if y < self.service_line_depth: return "Service Box (7)"
        if y < self.baseline_depth: return "Deep (8)"
        return "Very Deep (9)"