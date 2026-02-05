import torch
import numpy as np
import os
import sys

# Add MotionBERT root to sys.path to allow imports from lib.*
# Current file is in src/, MotionBERT is in MotionBERT-main/ relative to project root
MOTIONBERT_ROOT = '/kaggle/input/cv-auxiliary-repos/MotionBERT-main'
if MOTIONBERT_ROOT not in sys.path:
    sys.path.append(MOTIONBERT_ROOT)

class MotionBERTExtractor:
    def __init__(self, config_path=None, checkpoint_path=None, device='cuda'):
        self.device = device
        self.model = None
        
        print("Initializing MotionBERT Integrator...")
        
        # Attempt to load MotionBERT library
        try:
            from lib.model.DSTformer import DSTformer
            
            # Default args matching MB_ft_h36m.yaml or similar
            # These are the standard parameters for DSTformer used in MotionBERT
            # Based on configs/pose3d/MB_ft_h36m.yaml
            model_args = {
                'dim_in': 3, 
                'dim_out': 3, 
                'dim_feat': 512, 
                'dim_rep': 512, 
                'depth': 5, 
                'num_heads': 8, 
                'mlp_ratio': 2, 
                'att_fuse': True
            }
            
            self.model = DSTformer(**model_args)
            
            # DataParallel wrapper is used in MotionBERT scripts
            self.model = torch.nn.DataParallel(self.model)
            
            if checkpoint_path:
                # FIX: Handle directory input if user provides a folder instead of file
                if os.path.isdir(checkpoint_path):
                    print(f"Searching for checkpoint in directory: {checkpoint_path}")
                    candidates = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth') or f.endswith('.bin')]
                    if len(candidates) > 0:
                        checkpoint_path = os.path.join(checkpoint_path, candidates[0])
                        print(f"Found checkpoint: {checkpoint_path}")
                    else:
                        print(f"WARNING: No .pth/.bin files found in {checkpoint_path}")
                
                if os.path.exists(checkpoint_path):
                    print(f"Loading MotionBERT weights from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    
                    # Check formatting of checkpoint (e.g. 'model_pos', 'model', etc)
                    if 'model_pos' in checkpoint:
                         self.model.load_state_dict(checkpoint['model_pos'], strict=True)
                    else:
                         self.model.load_state_dict(checkpoint, strict=False)
                else:
                     print(f"WARNING: Checkpoint path invalid: {checkpoint_path}")
            else:
                print("WARNING: No checkpoint path provided for MotionBERT!")
            
            self.model.to(self.device)
            self.model.eval()
            self.valid = True

        except ImportError as e:
            print(f"ERROR: MotionBERT library import failed: {e}")
            print(f"Ensure that {MOTIONBERT_ROOT} exists and contains 'lib/model/DSTformer.py'")
            self.valid = False
        except Exception as e:
            print(f"ERROR initializing MotionBERT: {e}")
            import traceback
            traceback.print_exc()
            self.valid = False

    def _coco_to_h36m(self, kpts_coco):
        """
        Convert COCO (17 joints) to H36M (17 joints).
        Standard Human3.6M topology used by MotionBERT.
        """
        T, V, C = kpts_coco.shape
        kpts_h36m = np.zeros((T, 17, C), dtype=np.float32)
        
        # Mapping Implementation
        # H36M: 0:Pelvis, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle
        #       7:Spine, 8:Neck, 9:Nose, 10:Head, 11:LShoulder ...
        
        kpts_h36m[:, 0] = (kpts_coco[:, 11] + kpts_coco[:, 12]) * 0.5 # Pelvis
        kpts_h36m[:, 1] = kpts_coco[:, 12] # RHip
        kpts_h36m[:, 2] = kpts_coco[:, 14] # RKnee
        kpts_h36m[:, 3] = kpts_coco[:, 16] # RAnkle
        kpts_h36m[:, 4] = kpts_coco[:, 11] # LHip
        kpts_h36m[:, 5] = kpts_coco[:, 13] # LKnee
        kpts_h36m[:, 6] = kpts_coco[:, 15] # LAnkle
        
        kpts_h36m[:, 8] = (kpts_coco[:, 5] + kpts_coco[:, 6]) * 0.5 # Neck/Thorax
        kpts_h36m[:, 7] = (kpts_h36m[:, 0] + kpts_h36m[:, 8]) * 0.5 # Spine
        
        kpts_h36m[:, 9] = kpts_coco[:, 0] # Nose
        
        # Head (10)
        if C > 2 and (kpts_coco[:, 1, 2] > 0.1).any():
             kpts_h36m[:, 10] = (kpts_coco[:, 1] + kpts_coco[:, 2]) * 0.5
        else:
             kpts_h36m[:, 10] = kpts_coco[:, 0] # Fallback
             
        kpts_h36m[:, 11] = kpts_coco[:, 5] # LShoulder
        kpts_h36m[:, 12] = kpts_coco[:, 7] # LElbow
        kpts_h36m[:, 13] = kpts_coco[:, 9] # LWrist
        kpts_h36m[:, 14] = kpts_coco[:, 6] # RShoulder
        kpts_h36m[:, 15] = kpts_coco[:, 8] # RElbow
        kpts_h36m[:, 16] = kpts_coco[:, 10] # RWrist
        
        return kpts_h36m


    def _h36m_to_coco(self, kpts_h36m, original_coco=None):
        """
        Convert H36M (17 joints) back to COCO (17 joints).
        This is critical to maintain compatibility with GCNs trained on COCO topology.
        
        MotionBERT indices:
        0:Pelvis, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle
        7:Spine, 8:Neck, 9:Nose, 10:Head, 11:LShoulder, 12:LElbow, 
        13:LWrist, 14:RShoulder, 15:RElbow, 16:RWrist
        
        COCO Target indices:
        0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder
        7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip
        13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
        """
        T, V, C = kpts_h36m.shape
        kpts_coco = np.zeros((T, 17, C), dtype=np.float32)
        
        # Direct Mapping (Reverse of _coco_to_h36m)
        kpts_coco[:, 0] = kpts_h36m[:, 9] # Nose
        kpts_coco[:, 5] = kpts_h36m[:, 11] # L-Shoulder
        kpts_coco[:, 6] = kpts_h36m[:, 14] # R-Shoulder
        kpts_coco[:, 7] = kpts_h36m[:, 12] # L-Elbow
        kpts_coco[:, 8] = kpts_h36m[:, 15] # R-Elbow
        kpts_coco[:, 9] = kpts_h36m[:, 13] # L-Wrist
        kpts_coco[:, 10] = kpts_h36m[:, 16] # R-Wrist
        kpts_coco[:, 11] = kpts_h36m[:, 4] # L-Hip
        kpts_coco[:, 12] = kpts_h36m[:, 1] # R-Hip
        kpts_coco[:, 13] = kpts_h36m[:, 5] # L-Knee
        kpts_coco[:, 14] = kpts_h36m[:, 2] # R-Knee
        kpts_coco[:, 15] = kpts_h36m[:, 6] # L-Ankle
        kpts_coco[:, 16] = kpts_h36m[:, 3] # R-Ankle
        
        # Missing Joints in H36M (Eyes, Ears)
        # We need to approximate or use original COCO 2D data (with Z=0 or interpolated Z)
        # Since H36M has Head(10) and Nose(9), we can try to guess eyes.
        # But simpler is to use Nose Z for Eyes/Ears if we don't have original data.
        
        # For Z-axis (Index 2), copy Nose Z to Eyes/Ears
        nose_z = kpts_h36m[:, 9, 2:3] # (T, 1)
        
        kpts_coco[:, 1:5] = kpts_coco[:, 0:1] # Copy Nose X,Y,Z to Eyes/Ears temporarily
        kpts_coco[:, 1:5, 2:3] = nose_z[:, None, :] # Ensure Z is Nose Z (Fixed broadcasting)
        
        # If we have original COCO 2D input, we should restore X,Y for Eyes/Ears
        # because H36M doesn't track them at all.
        if original_coco is not None:
            # Indices 1,2,3,4 (Eyes, Ears)
            # Restore X,Y from original detection (more accurate than Nose approximation)
            kpts_coco[:, 1:5, :2] = original_coco[:, 1:5, :2]
            
        return kpts_coco

    def lift_2d_to_3d(self, keypoints_2d, image_size=None):
        """
        Lifts a sequence of 2D keypoints (COCO-17) to 3D (H36M-17).
        Args:
            keypoints_2d: Numpy array (T, 17, 2) or (T, 17, 3) [x, y, conf]
                          format: COCO
            image_size: (w, h) tuple, optional. Used for normalization.
                        If None, uses person bounding box for normalization.
        Returns:
            keypoints_3d: (T, 17, 3) [x, y, z]. 
                          IMPORTANT: Returns H36M format!
                          Use lift_2d_to_3d_coco if you need COCO format output.
        """
        if not self.valid:
            # Fallback or dummy return if model is missing
            return self._dummy_3d(keypoints_2d)

        # 1. Conversion COCO-17 -> H36M-17
        kpts_h36m = self._coco_to_h36m(keypoints_2d)
        
        # 2. Normalization
        # Model expects inputs normalized to [-1, 1] approximately
        # (x - w/2) / (w/2) is typical
        kpts_norm, params = self._normalize_for_motionbert(kpts_h36m, image_size)

        with torch.no_grad():
            # Prepare Input
            input_tensor = torch.tensor(kpts_norm, dtype=torch.float32).to(self.device)
            
            # Input format: (B, T, V, C)
            if input_tensor.dim() == 3: 
                input_tensor = input_tensor.unsqueeze(0)
            
            # Pass only (x, y) if model expects C=2, or (x, y, conf)
            # DSTformer usually takes 3 channels (x, y, score) or 2.
            # Assuming trained on 2D+Score or just 2D. 
            # Existing extractor passed everything. Let's keep C=3 if available, else C=2.
            if input_tensor.shape[-1] > 3:
                 input_tensor = input_tensor[..., :3]

            # Run Inference
            predicted_3d = self.model(input_tensor)
            
            if isinstance(predicted_3d, list):
                predicted_3d = predicted_3d[-1]
            
            output = predicted_3d[0].cpu().numpy() # (T, V, 3)
            
            # 3. Denormalization (Essential for mixing with original COCO data)
            factor = params['scale']
            center_x, center_y = params['center']
            
            # output is (T, V, 3)
            output[:, :, 0] = output[:, :, 0] * factor + center_x
            output[:, :, 1] = output[:, :, 1] * factor + center_y
            output[:, :, 2] = output[:, :, 2] * factor # Depth is relative scale
            
            return output
            
    def lift_2d_to_3d_coco(self, keypoints_2d, image_size=None):
        """
        Lifts to 3D and converts back to COCO topology.
        Safe for use with GCNs trained on COCO.
        """
        kpts_h36m_3d = self.lift_2d_to_3d(keypoints_2d, image_size)
        
        # Convert H36M 3D back to COCO 3D
        kpts_coco_3d = self._h36m_to_coco(kpts_h36m_3d, original_coco=keypoints_2d)
        
        return kpts_coco_3d

    def _normalize_for_motionbert(self, kpts, image_size=None):
        """
        Normalize keypoints to roughly [-1, 1].
        """
        T, V, C = kpts.shape
        kpts_norm = kpts.copy()
        
        # If image size is not provided, estimate from data range
        if image_size is None:
            # Use bounding box of the sequence
            min_x, max_x = np.min(kpts[:, :, 0]), np.max(kpts[:, :, 0])
            min_y, max_y = np.min(kpts[:, :, 1]), np.max(kpts[:, :, 1])
            w = max_x - min_x
            h = max_y - min_y
            center_x = min_x + w/2
            center_y = min_y + h/2
            scale = max(w, h)
        else:
            w_img, h_img = image_size
            center_x, center_y = w_img/2, h_img/2
            scale = min(w_img, h_img)
        
        factor = scale / 2.0 if image_size else scale
        
        kpts_norm[:, :, 0] = (kpts[:, :, 0] - center_x) / factor
        kpts_norm[:, :, 1] = (kpts[:, :, 1] - center_y) / factor
        
        return kpts_norm, {'center': (center_x, center_y), 'scale': factor}
    
    def _dummy_3d(self, kpts):
        """Helper to return 2D as 3D (z=0) if model fails"""
        T, V = kpts.shape[:2]
        out = np.zeros((T, V, 3))
        out[:, :, :2] = kpts[:, :, :2]
        return out


class MotionBERTIntegratedExtractor:
    """
    Integrates YOLO (via PoseExtractor) + MotionBERT Lifting.
    """
    def __init__(self, yolo_path, motionbert_ckpt, device='cuda'):
        from src.extractor import PoseExtractor
        
        self.pose_extractor = PoseExtractor(model_path=yolo_path, device=device)
        self.lifter = MotionBERTExtractor(checkpoint_path=motionbert_ckpt, device=device)
        
    def extract_sequence(self, video_path):
        # 1. Get 2D Keypoints (T, 17, 3) -> x, y, conf
        kpts_2d = self.pose_extractor.extract_sequence(video_path)
        
        if kpts_2d is None or len(kpts_2d) == 0:
            return None
            
        if isinstance(kpts_2d, list):
            kpts_2d = np.array(kpts_2d)
        
        # 2. Lift to 3D
        # Pass (x, y, score) or just (x, y) depending on training
        # Usually passing all 3 channels is safer if model handles it
        kpts_3d = self.lifter.lift_2d_to_3d_coco(kpts_2d)
        
        # 3. Merge Confidence
        conf = kpts_2d[:, :, 2:3] # (T, V, 1)
        kpts_final = np.concatenate([kpts_3d, conf], axis=2) # (T, V, 4) -> x, y, z, conf
        
        return kpts_final
