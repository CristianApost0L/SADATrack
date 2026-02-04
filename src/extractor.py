import cv2
import numpy as np
import torch
from ultralytics import YOLO
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

class PoseExtractor:
    SEQ_LEN = 40
    NUM_JOINTS = 17
    CONFIDENCE_THRESH = 0.25
    
    def __init__(self, model_path, device=None, seq_len=None, num_joints=None, confidence_thresh=None, smart_crop=False, detector_path=None, courtside_path=None):
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
            
        # Smart Crop Configuration
        self.smart_crop = smart_crop
        
        # 1. Main Detector (Person)
        det_path = detector_path if detector_path else 'yolov8n.pt'
        print(f"Loading Person Detector from {det_path} on {self.device}...")
        self.detector = YOLO(det_path)

        # 2. Racket Detector (CourtSide)
        # Using a distinct attribute to avoid confusion
        r_path = courtside_path if courtside_path else 'Davidsv/CourtSide-Computer-Vision-v1'
        
        # Check if it is the specific HF repo and try to download
        if r_path == 'Davidsv/CourtSide-Computer-Vision-v1':
            if hf_hub_download:
                try:
                    print(f"Downloading/Verifying Racket Detector from HuggingFace Hub: {r_path}")
                    r_path = hf_hub_download(repo_id=r_path, filename="model.pt")
                except Exception as e:
                    print(f"Error downloading from HF: {e}")
            else:
                 print("Warning: huggingface_hub not installed, cannot download model.")

        print(f"Loading Racket Detector from {r_path}...")
        try:
            self.racket_detector = YOLO(r_path)
            self.has_racket_detector = True
        except Exception as e:
            print(f"WARNING: Could not load CourtSide model: {e}")
            print("Falling back to standard Person Detection (Max Area).")
            self.has_racket_detector = False
            self.racket_detector = None
            
        print(f"Loading Pose model from {model_path} on {self.device}...")
        self.model = YOLO(model_path)

    def extract_sequence(self, video_path, num_players=1):
        """
        Extracts sequence for multiple players.
        Returns: (T, M, V, C) where M=num_players.
        Players are filtered by Racket ownership if available.
        """
        cap = cv2.VideoCapture(video_path)
        frames_data = [] # List of T frames, each containing M skeletons
        
        if not cap.isOpened(): return None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Init frame data: (M, V, C)
            current_frame_kpts = np.zeros((num_players, self.NUM_JOINTS, 3), dtype=np.float32)
            
            # 1. Detect Persons
            # Class 0 is 'person' in standard COCO models
            det_results = self.detector(frame, verbose=False, classes=[0], conf=self.CONFIDENCE_THRESH)
            
            # 2. Detect Rackets (if enabled)
            racket_boxes = None
            if self.has_racket_detector:
                # Assuming class 0 is racket in CourtSide model (or whatever class it detects)
                # We do not filter by class here to be safe, assuming model is specific
                try:
                    racket_res = self.racket_detector(frame, verbose=False, conf=0.15)
                    if racket_res and len(racket_res[0].boxes) > 0:
                        racket_boxes = racket_res[0].boxes
                except Exception as e:
                    # If inference fails for some reason, continue without racket
                    pass
            
            if det_results and len(det_results[0].boxes) > 0:
                person_boxes = det_results[0].boxes
                
                # 3. Select Targets (Racket-based + Size-based)
                target_indices = self._select_targets_with_racket(person_boxes, racket_boxes, num_targets=num_players)
                
                for idx_m, box_idx in enumerate(target_indices):
                    # --- SMART CROP LOGIC applied to each target ---
                    process_frame = frame
                    offset_x, offset_y = 0, 0
                    
                    # Get Box Info
                    box_xyxy = person_boxes.xyxy[box_idx].cpu().numpy()
                    x1, y1, x2, y2 = box_xyxy
                    
                    if self.smart_crop:
                        # Apply padding/crop
                        h, w = frame.shape[:2]
                        box_w = x2 - x1
                        box_h = y2 - y1
                        pad_w = box_w * 0.2
                        pad_h = box_h * 0.2
                        
                        crop_x1 = max(0, int(x1 - pad_w))
                        crop_y1 = max(0, int(y1 - pad_h))
                        crop_x2 = min(w, int(x2 + pad_w))
                        crop_y2 = min(h, int(y2 + pad_h))
                        
                        process_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        offset_x, offset_y = crop_x1, crop_y1

                    # 4. Pose Inference on Target
                    # Be very permissive with conf in crop, we know there is a person
                    pose_results = self.model(process_frame, verbose=False, conf=0.1) 
                    
                    if pose_results and len(pose_results[0].boxes) > 0:
                        # In the crop, the person should be the main subject (max area)
                        p_areas = pose_results[0].boxes.xywh[:, 2] * pose_results[0].boxes.xywh[:, 3]
                        p_best_idx = torch.argmax(p_areas).item()
                        
                        kpts_raw = pose_results[0].keypoints.data[p_best_idx].cpu().numpy()
                        p_box_h = pose_results[0].boxes.xywh[p_best_idx, 3].item()
                        p_box_center = pose_results[0].boxes.xywh[p_best_idx, :2].cpu().numpy()
                        
                        # Remap to Global
                        kpts_raw[:, 0] += offset_x
                        kpts_raw[:, 1] += offset_y
                        p_box_center[0] += offset_x
                        p_box_center[1] += offset_y
                        
                        # Normalize
                        norm_kpts = self._normalize(kpts_raw, p_box_h, p_box_center)
                        current_frame_kpts[idx_m] = norm_kpts

            frames_data.append(current_frame_kpts)
        
        cap.release()
        return self._post_process(frames_data)

    def _select_targets_with_racket(self, person_boxes, racket_boxes, num_targets=2):
        """
        Selects players using Racket proximity filter.
        Logic:
        1. Calculate Score for each person: Area + (Bonus if near a racket)
        2. Sort by Score
        3. Keep top N
        4. Sort top N by Y-coordinate
        """
        # person_boxes.xywh: (N, 4)
        if len(person_boxes) == 0: return []
        
        p_xywh = person_boxes.xywh
        areas = p_xywh[:, 2] * p_xywh[:, 3]
        scores = areas.clone() # Base score is area
        
        # Apply Racket Bonus
        if racket_boxes is not None:
             r_centers = racket_boxes.xywh[:, :2] # (R, 2)
             p_centers = p_xywh[:, :2]            # (P, 2)
             
             # Calculate distances between all people and all rackets
             # (P, 1, 2) - (1, R, 2) -> (P, R, 2) -> norm -> (P, R)
             dists = torch.norm(p_centers.unsqueeze(1) - r_centers.unsqueeze(0), dim=2)
             
             # Find min distance to ANY racket for each person
             if dists.shape[1] > 0:
                 min_dists, _ = torch.min(dists, dim=1) # (P,)
                 
                 # Threshold for "holding" a racket (e.g., width of the person)
                 widths = p_xywh[:, 2]
                 has_racket = (min_dists < widths * 1.5) # Permissive threshold
                 
                 # Bonus: Double the score so they jump to top of list
                 max_area = torch.max(areas) if len(areas) > 0 else 1.0
                 scores[has_racket] += max_area * 10.0 # Huge bonus

        # Select Top K
        k = min(len(person_boxes), num_targets)
        _, top_indices = torch.topk(scores, k)
        
        # Sort spatially (Top-Down Y)
        top_y_centers = p_xywh[top_indices, 1]
        y_sort_idx = torch.argsort(top_y_centers)
        
        final_indices = top_indices[y_sort_idx]
        
        return final_indices.tolist()

    def _select_targets(self, boxes, num_targets=2):
        # Deprecated usually, but kept as backup or for non-racket logic
        return self._select_targets_with_racket(boxes, None, num_targets)

    def _normalize(self, kpts_raw, box_h, box_center_xy):
        """
        Normalizza centrando sulle anche e scalando per l'altezza del giocatore.
        Include il canale confidenza per HD-GCN/CTR-GCN.
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
        if not frames_data: return None
        data = np.array(frames_data, dtype=np.float32)
        target_len = self.SEQ_LEN
        
        if len(data) < target_len:
            padding = np.zeros((target_len - len(data), self.NUM_JOINTS, 3), dtype=np.float32)
            data = np.vstack((data, padding))
        elif len(data) > target_len:
            data = data[np.linspace(0, len(data)-1, target_len).astype(int)]
        return data