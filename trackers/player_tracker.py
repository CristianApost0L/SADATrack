from ultralytics import YOLO 
import cv2
import pickle
import numpy as np
import sys
import constants
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox, get_foot_position

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections, player_detection_court_margin, last_known_positions=None):
        """
        1. Uses constraints to find players in Frame 0.
        2. Follows those IDs.
        3. Only allows handover to a NEW ID if it matches the LOCATION *and* SIZE of the lost player.
        """
        # Step 1: Initial Selection
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(
            court_keypoints, 
            player_detections_first_frame, 
            player_detection_court_margin, 
            last_known_positions
        )
        
        filtered_player_detections = []
        
        # Track both POSITION and BOX SIZE (Height) to prevent switching to ball boys
        # Format: { track_id: {'pos': (x,y), 'height': h, 'width': w} }
        active_player_stats = {}
        
        # Initialize stats from Frame 0
        for track_id in chosen_players:
            if track_id in player_detections_first_frame:
                bbox = player_detections_first_frame[track_id]['bbox']
                active_player_stats[track_id] = {
                    'pos': get_center_of_bbox(bbox),
                    'height': bbox[3] - bbox[1],
                    'width': bbox[2] - bbox[0]
                }

        # Step 2: Process all frames
        for frame_idx, player_dict in enumerate(player_detections):
            filtered_player_dict = {}
            
            # A. Check currently "Approved" IDs
            for track_id in chosen_players:
                if track_id in player_dict:
                    # Player found! Update stats.
                    filtered_player_dict[track_id] = player_dict[track_id]
                    
                    bbox = player_dict[track_id]['bbox']
                    active_player_stats[track_id] = {
                        'pos': get_center_of_bbox(bbox),
                        'height': bbox[3] - bbox[1],
                        'width': bbox[2] - bbox[0]
                    }
            
            # B. Handle Lost IDs (Strict Handover)
            if len(filtered_player_dict) < len(chosen_players):
                
                missing_ids = [pid for pid in chosen_players if pid not in filtered_player_dict]
                strangers = [pid for pid in player_dict if pid not in chosen_players]
                
                for missing_id in missing_ids:
                    last_stats = active_player_stats.get(missing_id)
                    if last_stats is None: continue
                    
                    best_candidate = None
                    min_dist = float('inf')
                    
                    # Define Dynamic Thresholds based on the player's last known size
                    # A player won't instantly teleport more than 2x their body width
                    max_allowed_dist = max(100, last_stats['width'] * 5) 
                    
                    for stranger_id in strangers:
                        bbox = player_dict[stranger_id]['bbox']
                        stranger_pos = get_center_of_bbox(bbox)
                        stranger_height = bbox[3] - bbox[1]
                        
                        distance = measure_distance(last_stats['pos'], stranger_pos)
                        
                        # --- CONSTRAINT 1: DISTANCE ---
                        if distance > max_allowed_dist:
                            continue
                            
                        # --- CONSTRAINT 2: SIZE SIMILARITY (The Anti-Ballboy Fix) ---
                        # Ball boys in the back are usually smaller (perspective) or crouching.
                        # We only accept a new ID if its height is within 30% of the lost player.
                        height_ratio = abs(stranger_height - last_stats['height']) / last_stats['height']
                        
                        # If the size difference is > 30%, it's likely a different person/object
                        if height_ratio > 0.3:
                            continue
                            
                        if distance < min_dist:
                            min_dist = distance
                            best_candidate = stranger_id
                    
                    # If we found a VALID candidate
                    if best_candidate is not None:
                        # Perform the Swap
                        chosen_players.remove(missing_id)
                        chosen_players.append(best_candidate)
                        
                        filtered_player_dict[best_candidate] = player_dict[best_candidate]
                        
                        # Update stats
                        bbox = player_dict[best_candidate]['bbox']
                        active_player_stats[best_candidate] = {
                            'pos': get_center_of_bbox(bbox),
                            'height': bbox[3] - bbox[1],
                            'width': bbox[2] - bbox[0]
                        }
                        
                        # Remove from strangers list
                        strangers.remove(best_candidate)
                        del active_player_stats[missing_id]
                        
                        print(f"Frame {frame_idx}: Handover {missing_id} -> {best_candidate} (Dist: {min_dist:.1f}px, Height Diff: {height_ratio:.2%})")

            filtered_player_detections.append(filtered_player_dict)
            
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict, player_detection_court_margin, last_known_positions=None):
        # 1. Convert keypoints to numpy for easier calc
        court_kps = np.array(court_keypoints).reshape(-1, 2)
        
        # 2. Separate points into Top (far) and Bottom (close) to define the trapezoid
        avg_y = np.mean(court_kps[:, 1])
        top_half = court_kps[court_kps[:, 1] < avg_y]
        bottom_half = court_kps[court_kps[:, 1] > avg_y]
        
        valid_ids = []
        forced_ids = set()

        # --- NEW LOGIC: Force-Keep Players from Previous Clip ---
        if last_known_positions is not None:
            print("Filtering using Last Known Positions...")
            for p_num, pos in last_known_positions.items():
                target_pos = np.array(pos)
                min_dist = float('inf')
                best_id = None
                
                # Find the closest detection to this known position
                for track_id, player_data in player_dict.items():
                    bbox = player_data["bbox"]
                    # Use center of box for simple distance check
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    current_pos = np.array([cx, cy])
                    
                    dist = np.linalg.norm(current_pos - target_pos)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_id = track_id
                
                # If we found a match within a reasonable range (e.g. 200px), FORCE KEEP IT
                if best_id is not None and min_dist < 200:
                    valid_ids.append(best_id)
                    forced_ids.add(best_id)
                    print(f"  [Override] Force-keeping ID {best_id} (Matches P{p_num}, Dist: {min_dist:.1f}px)")

        # Safety check
        if len(top_half) > 0 and len(bottom_half) > 0:
            # Find the 4 corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right
            tl = top_half[np.argmin(top_half[:, 0])]
            tr = top_half[np.argmax(top_half[:, 0])]
            bl = bottom_half[np.argmin(bottom_half[:, 0])]
            br = bottom_half[np.argmax(bottom_half[:, 0])]

            for track_id, player_data in player_dict.items():
                # If we forced this ID, skip ALL margin checks.
                if track_id in forced_ids:
                    continue
                
                bbox = player_data["bbox"]
                
                # CRITICAL FIX: Use foot position (ground plane), not center
                # This ensures the perspective calculation matches the court lines
                px, py = get_foot_position(bbox) 
                
                # Calculate the X-limits of the court at this specific Y (depth)
                # Linear interpolation: x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                
                # Left Line Limit
                left_limit_x = tl[0] + (bl[0] - tl[0]) * (py - tl[1]) / (bl[1] - tl[1] + 1e-6)
                
                # Right Line Limit
                right_limit_x = tr[0] + (br[0] - tr[0]) * (py - tr[1]) / (br[1] - tr[1] + 1e-6)
                
                # --- NEW Y-CHECK ---
                min_y = tl[1] - player_detection_court_margin
                max_y = bl[1] + player_detection_court_margin

                if not (min_y < py < max_y):
                    print(f"[DEBUG] ID {track_id} removed: Y-pos {py:.1f} is outside depth limits [{min_y:.1f}, {max_y:.1f}]")
                    continue
                
                # Check if player is within the width limits
                margin = player_detection_court_margin
                
                if (left_limit_x - margin) < px < (right_limit_x + margin):
                    valid_ids.append(track_id)
                else:
                    print(f"[DEBUG] Filtered out ID {track_id}: X={px:.1f} is outside range [{left_limit_x - margin:.1f}, {right_limit_x + margin:.1f}]")
        
        # Fallback: If strict filtering removed everyone, revert to all
        if len(valid_ids) < 2:
            print("[WARNING] Spatial filter removed too many players. Reverting to distance-only check.")
            valid_ids = list(player_dict.keys())

        # Standard Distance Check (only on the valid IDs)
        distances = []
        for track_id in valid_ids:
            player_data = player_dict[track_id]
            bbox = player_data["bbox"]
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        distances.sort(key = lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, yolo_verbosity = False):
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame, yolo_verbosity)
            player_detections.append(player_dict)
        
        return player_detections

    def detect_frame(self, frame, yolo_verbosity = False):
        # persist = True to track
        results = self.model.track(frame, persist=True, verbose = yolo_verbosity)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            # --- FIX: Check if track ID exists ---
            if box.id is None:
                continue
                
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            
            if object_cls_name == "person":
                # Save BOTH bounding box AND keypoints
                # Check if keypoints exist (YOLO pose returns them in results.keypoints)
                keypoints = results.keypoints.data[list(results.boxes.id).index(track_id)].tolist() if results.keypoints is not None else []
                player_dict[track_id] = {"bbox": result, "keypoints": keypoints}
        
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, player_data in player_dict.items():
                # Extract bbox from the new dictionary structure
                bbox = player_data["bbox"] 
                
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames