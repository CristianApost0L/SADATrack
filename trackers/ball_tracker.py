import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from models.tracknet import BallTrackerNet

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
        """
        Run model on a list of consecutive video frames.
        Returns a list of (x, y) coordinates.
        """
        ball_track = [(None, None)]*2
        prev_pred = [None, None]
        
        print("Running Ball Tracking...")
        for num in tqdm(range(2, len(frames))):
            # Resize frames to model input size
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            
            # Stack frames (9 channels total)
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            # Inference
            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
                
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
            
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
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
            # If we have a previous detection, use it to filter outliers
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                # If no previous detection, take the first/strongest circle
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        
        return x, y
    
    def interpolate_ball_positions(self, ball_track):
        # Helper to fill None values using pandas interpolation (from notebook logic)
        import pandas as pd
        df = pd.DataFrame(ball_track, columns=['x', 'y'])
        df = df.interpolate()
        return list(zip(df['x'], df['y']))