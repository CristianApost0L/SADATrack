import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PoseExtractor:
    """Pose Extractor focusing on the main foreground player with smart crop."""
    SEQ_LEN = 40
    NUM_JOINTS = 17
    CONFIDENCE_THRESH = 0.25
    
    def __init__(self, model_path, device=None, seq_len=None, num_joints=None, confidence_thresh=None, smart_crop=False):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Override parameters if provided
        if seq_len is not None:
            self.SEQ_LEN = seq_len
        if num_joints is not None:
            self.NUM_JOINTS = num_joints
        if confidence_thresh is not None:
            self.CONFIDENCE_THRESH = confidence_thresh
            
        # Smart Crop Configuration
        self.smart_crop = smart_crop
            
        print(f"Loading YOLO-Pose model from {model_path} on {self.device}...")
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        """
        Detect main player's keypoints in a single frame with optional smart crop.
        
        Args:
            frame: Input frame (H, W, 3) BGR image
            
        Returns:
            tuple: (kpts_raw, box_xyxy) where:
                - kpts_raw: (NUM_JOINTS, 3) raw keypoints in original frame coords
                - box_xyxy: (4,) bounding box [x1, y1, x2, y2] or None if no detection
        """
        # Initial detection
        results = self.model(frame, verbose=False, conf=self.CONFIDENCE_THRESH)
        
        if not results or len(results[0].boxes) == 0:
            return None, None
        
        # Select the largest person (main foreground player)
        boxes = results[0].boxes
        areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]
        best_idx = torch.argmax(areas).item()
        
        # Extract keypoints and box info
        kpts_raw = results[0].keypoints.data[best_idx].cpu().numpy()
        box_xyxy = boxes.xyxy[best_idx].cpu().numpy()
        
        # Smart Crop: Re-run on cropped region for better resolution
        if self.smart_crop:
            x1, y1, x2, y2 = box_xyxy
            h, w = frame.shape[:2]
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Add 20% padding
            pad_w = box_w * 0.2
            pad_h = box_h * 0.2
            
            crop_x1 = max(0, int(x1 - pad_w))
            crop_y1 = max(0, int(y1 - pad_h))
            crop_x2 = min(w, int(x2 + pad_w))
            crop_y2 = min(h, int(y2 + pad_h))
            
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_results = self.model(cropped, verbose=False, conf=0.1)
            
            if crop_results and len(crop_results[0].boxes) > 0:
                # Use the largest detection in crop
                crop_areas = crop_results[0].boxes.xywh[:, 2] * crop_results[0].boxes.xywh[:, 3]
                crop_best = torch.argmax(crop_areas).item()
                
                kpts_raw = crop_results[0].keypoints.data[crop_best].cpu().numpy()
                crop_box_xyxy = crop_results[0].boxes.xyxy[crop_best].cpu().numpy()
                
                # Remap coordinates to original frame
                kpts_raw[:, 0] += crop_x1
                kpts_raw[:, 1] += crop_y1
                box_xyxy = crop_box_xyxy + [crop_x1, crop_y1, crop_x1, crop_y1]
        
        return kpts_raw, box_xyxy

    def extract_sequence(self, video_path):
        """
        Extracts pose sequence from video, focusing on the main foreground player.
        Returns raw keypoints - normalization should be applied separately.
        
        Returns: (T, V, C) where T=frames, V=joints, C=channels (x,y,conf) - RAW coordinates
        """
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        if not cap.isOpened(): 
            return None
        
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Initialize frame skeleton
            current_frame_kpts = np.zeros((self.NUM_JOINTS, 3), dtype=np.float32)
            
            # Detect keypoints using shared detection method
            kpts_raw, box_xyxy = self.detect_frame(frame)
            
            if kpts_raw is not None:
                # Store raw keypoints
                current_frame_kpts = kpts_raw
            
            frames_data.append(current_frame_kpts)
        
        cap.release()
        return self._post_process(frames_data)

    def _post_process(self, frames_data):
        """Resample or pad sequence to target length."""
        if not frames_data: 
            return None
        
        data = np.array(frames_data, dtype=np.float32)
        target_len = self.SEQ_LEN
        
        if len(data) < target_len:
            # Pad with zeros
            padding = np.zeros((target_len - len(data), self.NUM_JOINTS, 3), dtype=np.float32)
            data = np.vstack((data, padding))
        elif len(data) > target_len:
            # Resample uniformly
            indices = np.linspace(0, len(data)-1, target_len).astype(int)
            data = data[indices]
        
        return data