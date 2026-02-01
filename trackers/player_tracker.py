from ultralytics import YOLO 
import cv2
import pickle
import numpy as np
import sys
import constants
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            # Preserve the whole data object (bbox + keypoints) for chosen players
            filtered_player_dict = {track_id: player_data for track_id, player_data in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        # --- NEW FILTERING LOGIC ---
        # 1. Convert keypoints to a numpy array for easier calculation
        court_kps = np.array(court_keypoints).reshape(-1, 2)
        
        # 2. Separate points into Top (far) and Bottom (close) clusters to define the trapezoid
        avg_y = np.mean(court_kps[:, 1])
        top_half = court_kps[court_kps[:, 1] < avg_y]
        bottom_half = court_kps[court_kps[:, 1] > avg_y]
        
        # 3. Define the Valid ID list
        valid_ids = []
        
        # Check if we have enough points to define lines (Safety check)
        if len(top_half) > 0 and len(bottom_half) > 0:
            # Find the corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right
            tl = top_half[np.argmin(top_half[:, 0])]
            tr = top_half[np.argmax(top_half[:, 0])]
            bl = bottom_half[np.argmin(bottom_half[:, 0])]
            br = bottom_half[np.argmax(bottom_half[:, 0])]

            for track_id, player_data in player_dict.items():
                bbox = player_data["bbox"]
                px, py = get_center_of_bbox(bbox)
                
                # 4. Calculate the expected Left and Right X coordinates at the player's Y position
                # Using linear interpolation between the Top and Bottom corners
                # Formula: x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                
                # Left Limit (from Top-Left to Bottom-Left)
                left_limit_x = tl[0] + (bl[0] - tl[0]) * (py - tl[1]) / (bl[1] - tl[1] + 1e-6)
                
                # Right Limit (from Top-Right to Bottom-Right)
                right_limit_x = tr[0] + (br[0] - tr[0]) * (py - tr[1]) / (br[1] - tr[1] + 1e-6)
                
                # 5. Check if player is between the lines (with a generous margin for running wide)
                margin = constants.COURT_MARGIN_FOR_PLAYER_DETECTION # pixels
                if (left_limit_x - margin) < px < (right_limit_x + margin):
                    valid_ids.append(track_id)
        
        # Fallback: If filter removes everyone (or too many), revert to checking all
        if len(valid_ids) < 2:
            valid_ids = list(player_dict.keys())
        # ---------------------------

        distances = []
        # Update loop to iterate only over 'valid_ids'
        for track_id in valid_ids:
            player_data = player_dict[track_id]
            
            # Extract bbox from the new dictionary structure
            bbox = player_data["bbox"]
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
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

    def detect_frame(self, frame, yolo_verbosity):
        # Change persist=True to track
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