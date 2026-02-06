import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PoseExtractor:
    """Pose Extractor focusing on the main foreground player."""
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

    def extract_sequence(self, video_path):
        """
        Extracts pose sequence from video, focusing on the main foreground player.
        Returns: (T, V, C) where T=frames, V=joints, C=channels (x,y,conf)
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
            
            # Detect person with YOLO-Pose (detects person + keypoints in one pass)
            results = self.model(frame, verbose=False, conf=self.CONFIDENCE_THRESH)
            
            if results and len(results[0].boxes) > 0:
                # Select the largest person (main foreground player)
                boxes = results[0].boxes
                areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]
                best_idx = torch.argmax(areas).item()
                
                # Extract keypoints and box info
                kpts_raw = results[0].keypoints.data[best_idx].cpu().numpy()
                box_h = boxes.xywh[best_idx, 3].item()
                box_center = boxes.xywh[best_idx, :2].cpu().numpy()
                box_xyxy = boxes.xyxy[best_idx].cpu().numpy()
                
                # Smart Crop: Re-run on cropped region for better resolution
                if self.smart_crop:
                    x1, y1, x2, y2 = box_xyxy
                    h, w = frame.shape[:2]
                    box_w = x2 - x1
                    box_h_box = y2 - y1
                    
                    # Add 20% padding
                    pad_w = box_w * 0.2
                    pad_h = box_h_box * 0.2
                    
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
                        box_h = crop_results[0].boxes.xywh[crop_best, 3].item()
                        box_center = crop_results[0].boxes.xywh[crop_best, :2].cpu().numpy()
                        
                        # Remap coordinates to original frame
                        kpts_raw[:, 0] += crop_x1
                        kpts_raw[:, 1] += crop_y1
                        box_center[0] += crop_x1
                        box_center[1] += crop_y1
                
                # Normalize keypoints
                current_frame_kpts = self._normalize(kpts_raw, box_h, box_center)
            
            frames_data.append(current_frame_kpts)
        
        cap.release()
        return self._post_process(frames_data)

    def _normalize(self, kpts_raw, box_h, box_center_xy):
        """
        Normalize keypoints by centering on the root (hips) and scaling by box height.
        """
        norm_kpts = np.zeros_like(kpts_raw)
        xy, conf = kpts_raw[:, :2], kpts_raw[:, 2]
        
        # Centratura sulle anche (Root)
        if conf[11] > 0.1 and conf[12] > 0.1: root = (xy[11] + xy[12]) / 2.0
        elif conf[11] > 0.1: root = xy[11]
        elif conf[12] > 0.1: root = xy[12]
        else: root = box_center_xy
        
        scale = box_h if box_h > 0 else 1.0
        norm_kpts[:, :2] = (xy - root) / scale
        norm_kpts[:, 2] = conf # Fondamentale per la confidenza nativa
        return norm_kpts

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