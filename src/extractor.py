import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PoseExtractor:
    SEQ_LEN = 40
    NUM_JOINTS = 17
    CONFIDENCE_THRESH = 0.25
    
    def __init__(self, model_path, device=None, seq_len=None, num_joints=None, confidence_thresh=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Sovrascrivi i parametri se forniti
        if seq_len is not None:
            self.SEQ_LEN = seq_len
        if num_joints is not None:
            self.NUM_JOINTS = num_joints
        if confidence_thresh is not None:
            self.CONFIDENCE_THRESH = confidence_thresh
            
        print(f"Loading YOLO model from {model_path} on {self.device}...")
        self.model = YOLO(model_path)

    def extract_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_data = []
        
        if not cap.isOpened(): return None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_kpts = np.zeros((self.NUM_JOINTS, 3), dtype=np.float32)
            
            # Inference
            results = self.model(frame, verbose=False, conf=self.CONFIDENCE_THRESH)
            
            if results and len(results[0].boxes) > 0:
                # Logica Max Area
                areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
                best_idx = torch.argmax(areas).item()
                
                kpts_raw = results[0].keypoints.data[best_idx].cpu().numpy()
                box_h = results[0].boxes.xywh[best_idx, 3].item()
                
                # Normalizzazione
                frame_kpts = self._normalize(kpts_raw, box_h, box_center_xy=results[0].boxes.xywh[best_idx, :2].cpu().numpy())
            
            frames_data.append(frame_kpts)
        
        cap.release()
        return self._post_process(frames_data)

    def _normalize(self, kpts_raw, box_h, box_center_xy):
        norm_kpts = np.zeros_like(kpts_raw)
        
        xy = kpts_raw[:, :2]
        conf = kpts_raw[:, 2]
        
        # --- 1. FIND ROOT CENTER ---
        # In tennis, we use the hips as root center. COCO Keypoints: 11=Left Hip, 12=Right Hip.
        
        # If both hips are visible
        if conf[11] > 0.1 and conf[12] > 0.1:
            root_center = (xy[11] + xy[12]) / 2.0
            
        # If only one hip is visible
        elif conf[11] > 0.1:
            root_center = xy[11]
        elif conf[12] > 0.1:
            root_center = xy[12]
            
        # If no hips are visible, use box center
        else:
            root_center = box_center_xy

        # --- 2. CALCULATE SCALE FACTOR ---
        # Use box height as scale factor to maintain aspect ratio
        scale_factor = box_h if box_h > 0 else 1.0
        
        # --- 3. NORMALIZE KEYPOINTS ---
        norm_kpts[:, :2] = (xy - root_center) / scale_factor
        norm_kpts[:, 2] = conf
        
        return norm_kpts

    def _post_process(self, frames_data):

        data = np.array(frames_data, dtype=np.float32)

        if len(data) == 0 or np.sum(data) == 0:
            return None
            
        current_len = len(data)
        target_len = self.SEQ_LEN
        
        # If the video's length is less than SEQ_LEN, pad with zeros
        if current_len < target_len:
            padding_len = target_len - current_len
            padding = np.zeros((padding_len, self.NUM_JOINTS, 3), dtype=np.float32)
            data = np.vstack((data, padding))
            
        # If the video's length is more than SEQ_LEN, sample frames uniformly
        elif current_len > target_len:
            indices = np.linspace(0, current_len - 1, target_len).astype(int)
            data = data[indices]
            
        return data