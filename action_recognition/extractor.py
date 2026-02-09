import numpy as np

class PoseExtractor:
    def __init__(self):
        self.SEQ_LEN = 40
        self.NUM_JOINTS = 17
        
    def process_sequence(self, sequence_data):
        """
        Processes a sequence of raw keypoints and bounding boxes.
        
        Args:
            sequence_data: List of dictionaries. Each dict represents a frame and must contain:
                           - 'keypoints': list or numpy array of shape (17, 3) [x, y, conf]
                           - 'bbox': list [x1, y1, x2, y2]
        Returns:
            np.array: Normalized sequence ready for the classifier (SEQ_LEN, 17, 3)
        """
        frames_data = []
        
        for frame_info in sequence_data:
            # SAFETY: Handle cases where keypoints are None or empty
            raw_kpts = frame_info.get('keypoints', [])
            bbox = frame_info.get('bbox', [0, 0, 1, 1])
            
            # Convert to numpy
            kpts_np = np.array(raw_kpts)
            
            # --- CRITICAL FIX ---
            # If the array is empty, 1D, or doesn't have 17 joints, use dummy data
            if kpts_np.size == 0 or kpts_np.ndim < 2 or kpts_np.shape[0] != 17:
                # Create a dummy zero-filled frame (17 joints, 3 values: x,y,conf)
                frame_kpts = np.zeros((17, 3), dtype=np.float32)
            else:
                # Valid data: Proceed with normalization
                x1, y1, x2, y2 = bbox
                box_h = y2 - y1
                box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                frame_kpts = self._normalize(kpts_np, box_h, box_center)
            
            frames_data.append(frame_kpts)
        
        return self._post_process(frames_data)

    def _normalize(self, kpts_raw, box_h, box_center_xy):
        """
        Normalizes keypoints to be invariant to scale and translation.
        Centers the skeleton on the hips and scales by box height.
        """
        norm_kpts = np.zeros_like(kpts_raw)
        
        xy = kpts_raw[:, :2]
        conf = kpts_raw[:, 2]
        
        # --- 1. FIND ROOT CENTER ---
        # In tennis, we use the hips as root center. COCO Keypoints: 11=Left Hip, 12=Right Hip.
        
        # If both hips are visible (confidence > 0.1)
        if conf[11] > 0.1 and conf[12] > 0.1:
            root_center = (xy[11] + xy[12]) / 2.0
            
        # If only one hip is visible
        elif conf[11] > 0.1:
            root_center = xy[11]
        elif conf[12] > 0.1:
            root_center = xy[12]
            
        # If no hips are visible, use the bounding box center as a fallback
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
        """
        Standardizes the sequence length to SEQ_LEN (40 frames).
        Pads with zeros if too short, samples uniformly if too long.
        """
        data = np.array(frames_data, dtype=np.float32)

        if len(data) == 0:
            # Return a zero block if data is empty to prevent crashes
            return np.zeros((self.SEQ_LEN, self.NUM_JOINTS, 3), dtype=np.float32)
            
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