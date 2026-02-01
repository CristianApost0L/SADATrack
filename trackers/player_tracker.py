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

    def choose_and_filter_players(self, court_keypoints, player_detections, player_detection_court_margin):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame, player_detection_court_margin)
        filtered_player_detections = []
        for player_dict in player_detections:
            # Preserve the whole data object (bbox + keypoints) for chosen players
            filtered_player_dict = {track_id: player_data for track_id, player_data in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict, player_detection_court_margin):
        # 1. Convert keypoints to numpy for easier calc
        court_kps = np.array(court_keypoints).reshape(-1, 2)
        
        # 2. Separate points into Top (far) and Bottom (close) to define the trapezoid
        avg_y = np.mean(court_kps[:, 1])
        top_half = court_kps[court_kps[:, 1] < avg_y]
        bottom_half = court_kps[court_kps[:, 1] > avg_y]
        
        valid_ids = []
        
        # Safety check
        if len(top_half) > 0 and len(bottom_half) > 0:
            # Find the 4 corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right
            tl = top_half[np.argmin(top_half[:, 0])]
            tr = top_half[np.argmax(top_half[:, 0])]
            bl = bottom_half[np.argmin(bottom_half[:, 0])]
            br = bottom_half[np.argmax(bottom_half[:, 0])]

            for track_id, player_data in player_dict.items():
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

    def detect_frames(self, frames, yolo_verbosity = False, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame, yolo_verbosity)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame, yolo_verbosity = False):
        # persist = True to track
        # augment = True to use TTA
        results = self.model.track(frame, persist=True, augment = True, verbose = yolo_verbosity)[0]
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