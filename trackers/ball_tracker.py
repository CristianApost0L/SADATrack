import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from tracknet.tracknet import BallTrackerNet

class BallTracker:
    def __init__(self, model_path, device='cuda'):
        # Input channels=9 (3 frames * 3 RGB), Out channels=256 (Heatmap depth)
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            
        # Model expects this input resolution
        self.width = 640
        self.height = 360

    def detect_frames(self, frames):
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        
        # 1. Get Original Dimensions
        original_h, original_w = frames[0].shape[:2]
        
        # 2. Calculate Scale Factors
        scale_x = original_w / self.width
        scale_y = original_h / self.height

        print("Running Ball Tracking...")
        for num in tqdm(range(2, len(frames))):
            # Resize 3 consecutive frames
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            
            # Stack frames: (Height, Width, 9) -> (3 frames * 3 channels)
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            
            # Normalize (0-255 -> 0-1)
            imgs = imgs.astype(np.float32) / 255.0
            
            # Transpose to PyTorch format: (Channels, Height, Width)
            imgs = np.rollaxis(imgs, 2, 0)
            
            # Add Batch Dimension: (1, 9, 360, 640)
            inp = np.expand_dims(imgs, axis=0)

            # Inference
            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                
            # Post-process with dynamic scales
            x_pred, y_pred = self.postprocess(output, prev_pred, scale_x, scale_y)
            
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
            
        return ball_track
    
    def get_ball_shot_frames(self, ball_positions):
        """
        Identifies frames where the ball changes direction vertically (hits or bounces).
        main.py will later filter these to ensure a player is close enough to hit it.
        """
        ball_shot_frames = []
        
        # We need a small buffer to detect direction change
        for i in range(1, len(ball_positions)-1):
            curr_pos = ball_positions[i]
            prev_pos = ball_positions[i-1]
            next_pos = ball_positions[i+1]
            
            # Skip if tracking was lost in any of these frames
            if curr_pos is None or prev_pos is None or next_pos is None:
                continue
            
            # Check only the Y coordinates (vertical movement)
            # Tuple structure is (x, y)
            y_prev = prev_pos[1]
            y_curr = curr_pos[1]
            y_next = next_pos[1]
            
            # DETECT PEAK OR VALLEY
            # Case 1: Ball was going down, now going up (Valley)
            if y_prev < y_curr and y_curr > y_next:
                ball_shot_frames.append(i)
                
            # Case 2: Ball was going up, now going down (Peak)
            elif y_prev > y_curr and y_curr < y_next:
                ball_shot_frames.append(i)

        return ball_shot_frames

    def postprocess(self, feature_map, prev_pred, scale_x, scale_y, max_dist=80):
        """
        Extracts ball coordinates from the heatmap using HoughCircles.
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
                                   param1=50, param2=2, minRadius=2, maxRadius=7)
        x, y = None, None
        
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    # Apply specific X and Y scales
                    x_temp = circles[0][i][0] * scale_x
                    y_temp = circles[0][i][1] * scale_y
                    
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0] * scale_x
                y = circles[0][0][1] * scale_y
        
        return x, y
    
    def interpolate_ball_positions(self, ball_track):
        # Helper to fill None values using pandas interpolation (from notebook logic)
        import pandas as pd
        df = pd.DataFrame(ball_track, columns=['x', 'y'])
        df = df.interpolate()
        return list(zip(df['x'], df['y']))
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw the ball if it exists in this frame
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # Only draw if box is valid (not 0,0,0,0)
                if x1 == 0 and x2 == 0:
                    continue
                    
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]), int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames