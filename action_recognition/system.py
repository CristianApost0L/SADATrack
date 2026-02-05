import torch
import numpy as np
import os, sys
import constants

mb_path = os.path.join(constants.PATH_FOR_AUXILIARY_DATASETS, 'MotionBERT-main')
if mb_path not in sys.path:
    sys.path.append(mb_path)

# 2. Add CTR-GCN path
ctr_path = os.path.join(constants.PATH_FOR_AUXILIARY_DATASETS, 'CTR-GCN-main')
if ctr_path not in sys.path:
    sys.path.append(ctr_path)

from .model import CTRGCN_Tennis
from .dataset import COCO_BONE_PAIRS
from .motionbert_extractor import MotionBERTExtractor


class Tennis3DSystem:
    def __init__(self, joint_weights, bone_weights, motionbert_weights, device='cuda'):
        self.device = device
        print(f"[3D System] Initializing on {device}...")

        # 1. Initialize MotionBERT (The "True 3D" Lifter)
        self.lifter = MotionBERTExtractor(checkpoint_path=motionbert_weights, device=device)

        # 2. Initialize Joint Model (CTRGCN)
        self.model_joint = CTRGCN_Tennis(num_classes=12, in_channels=3) # 3 channels: X, Y, Z
        self.model_joint.load_state_dict(torch.load(joint_weights, map_location=device))
        self.model_joint.to(device).eval()

        # 3. Initialize Bone Model (CTRGCN)
        self.model_bone = CTRGCN_Tennis(num_classes=12, in_channels=3)
        self.model_bone.load_state_dict(torch.load(bone_weights, map_location=device))
        self.model_bone.to(device).eval()

    def predict(self, kpts_2d_sequence):
        """
        Takes 2D Keypoints -> Lifts to 3D -> Generates Bones -> Ensemble Prediction
        Args:
            kpts_2d_sequence: (40, 17, 3) array [x, y, conf]
        """
        # --- A. LIFT TO 3D ---
        # MotionBERT expects (T, 17, 3). We ignore confidence for lifting usually, 
        # or MotionBERT handles it. The extractor returns (T, 17, 3) in 3D space.
        kpts_3d = self.lifter.extract_sequence(kpts_2d_sequence) # Returns (40, 17, 3)

        # Prepare for Pytorch: (Batch, Channel, Time, Vertex, Person)
        # Shape: (1, 3, 40, 17, 1)
        data_joint = torch.tensor(kpts_3d, dtype=torch.float32).to(self.device)
        data_joint = data_joint.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)

        # --- B. CREATE BONE DATA ---
        data_bone = torch.zeros_like(data_joint)
        for v1, v2 in COCO_BONE_PAIRS:
            data_bone[:, :, :, v1, :] = data_joint[:, :, :, v1, :] - data_joint[:, :, :, v2, :]

        # --- C. INFERENCE ---
        with torch.no_grad():
            output_joint = self.model_joint(data_joint)
            output_bone = self.model_bone(data_bone)

            # Ensemble: Average the logits
            final_output = (output_joint + output_bone) / 2
            probs = torch.softmax(final_output, dim=1)

        return probs.cpu().numpy()[0]