import torch
import numpy as np
import os
import sys

# Add MotionBERT root to sys.path to allow imports from lib.*
# Current file is in src/, MotionBERT is in MotionBERT-main/ relative to project root
MOTIONBERT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MotionBERT-main'))
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

    def lift_2d_to_3d(self, keypoints_2d):
        """
        Lifts a sequence of 2D keypoints to 3D.
        Args:
            keypoints_2d: Numpy array (T, V, 2) or (T, V, 3) [x, y, conf]
                          MotionBERT expects (B, T, V, C) where C=2 or 3.
        Returns:
            keypoints_3d: (T, V, 3) [x, y, z]
        """
        if not self.valid:
            # Fallback or dummy return if model is missing
            return self._dummy_3d(keypoints_2d)

        with torch.no_grad():
            # Prepare Input
            input_tensor = torch.tensor(keypoints_2d, dtype=torch.float32).to(self.device)
            
            # Input format expected: (B, T, V, C)
            if input_tensor.dim() == 3: # (T, V, C)
                input_tensor = input_tensor.unsqueeze(0) # (1, T, V, C)
            
            # Ensure C=2 or C=3. If C=3 (x,y,conf), typically we pass (x,y) if model expects 2D,
            # or (x,y,conf) if model expects it.
            # MotionBERT Lite usually takes (x,y) or (x,y,prob).
            
            # Run Inference
            predicted_3d = self.model(input_tensor)
            
            # Output handling
            # DSTformer forward returns the 3D sequence
            # Output shape: (B, T, V, 3)
            
            if isinstance(predicted_3d, list):
                predicted_3d = predicted_3d[-1]
            
            output = predicted_3d[0].cpu().numpy() # (T, V, 3)
            
            return output

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
        kpts_3d = self.lifter.lift_2d_to_3d(kpts_2d)
        
        # 3. Merge Confidence
        conf = kpts_2d[:, :, 2:3] # (T, V, 1)
        kpts_final = np.concatenate([kpts_3d, conf], axis=2) # (T, V, 4) -> x, y, z, conf
        
        return kpts_final
