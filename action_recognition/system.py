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

from .model import HDGCN_Tennis
from .dataset import COCO_BONE_PAIRS, normalize_skeleton
from .motionbert_extractor import MotionBERTExtractor


class Tennis3DSystem:
    def __init__(self, joint_weights, bone_weights, motionbert_weights, device='cuda'):
        self.device = device
        print(f"[3D System] Initializing on {device}...")

        # 1. Initialize MotionBERT (The 3D Lifter)
        self.lifter = MotionBERTExtractor(checkpoint_path=motionbert_weights, device=device)

        # 2. Initialize Joint Model (HDGCN)
        # Note: in_channels=4 because your weights expect (X, Y, Z, Confidence)
        print(f"Loading HDGCN Joint Model from {joint_weights}...")
        self.model_joint = HDGCN_Tennis(num_classes=12, in_channels=4) 
        self.model_joint.load_state_dict(torch.load(joint_weights, map_location=device))
        self.model_joint.to(device).eval()

        # 3. Initialize Bone Model (HDGCN)
        print(f"Loading HDGCN Bone Model from {bone_weights}...")
        self.model_bone = HDGCN_Tennis(num_classes=12, in_channels=4)
        self.model_bone.load_state_dict(torch.load(bone_weights, map_location=device))
        self.model_bone.to(device).eval()

    def predict(self, kpts_2d_sequence):
        """
        Args:
            kpts_2d_sequence: (40, 17, 3) array [x, y, conf]
        """
        # --- A. LIFT TO 3D ---
        # MotionBERT takes 2D and gives us 3D positions (X, Y, Z)
        # Output shape: (40, 17, 3)
        kpts_3d = self.lifter.lift_2d_to_3d_coco(kpts_2d_sequence) 

        # --- B. COMBINE CHANNELS (The Critical Fix) ---
        # Your HDGCN weights expect 4 channels: (X, Y, Z, Confidence).
        # MotionBERT gives (X, Y, Z). YOLO gives (Confidence). We must merge them.
        
        # 1. Extract Confidence from the original 2D input (Index 2)
        conf_channel = kpts_2d_sequence[:, :, 2:3] # Shape (40, 17, 1)
        
        # 2. Concatenate: (40, 17, 3) + (40, 17, 1) -> (40, 17, 4)
        kpts_combined = np.concatenate([kpts_3d, conf_channel], axis=2)

        # --- B. NORMALIZE (THE MISSING STEP) ---
        # We must align the skeleton (center hips, rotate view) just like in training.
        # normalize_skeleton expects (T, V, C) or (C, T, V) and returns (C, T, V)
        kpts_norm = normalize_skeleton(kpts_combined) # Returns (4, 40, 17)

        kpts_norm[0, :, :] = -kpts_norm[0, :, :]

        # 3. Prepare for Pytorch: (Batch, Channel, Time, Vertex, Person)
        # Shape: (1, 4, 40, 17, 1)
        data_joint = torch.tensor(kpts_combined, dtype=torch.float32).to(self.device)
        data_joint = data_joint.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)

        # --- C. PREPARE TENSOR ---
        # Shape needed: (Batch, Channel, Time, Vertex, Person) -> (1, 4, 40, 17, 1)
        data_joint = torch.tensor(kpts_norm, dtype=torch.float32).to(self.device)
        data_joint = data_joint.unsqueeze(0).unsqueeze(-1) # Add Batch and Person dims

        # --- D. CREATE BONE DATA ---
        data_bone = torch.zeros_like(data_joint)
        for v1, v2 in COCO_BONE_PAIRS:
            data_bone[:, :, :, v1, :] = data_joint[:, :, :, v1, :] - data_joint[:, :, :, v2, :]

        # --- E. INFERENCE ---
        with torch.no_grad():
            output_joint = self.model_joint(data_joint)
            output_bone = self.model_bone(data_bone)

            final_output = (output_joint + output_bone) / 2
            probs = torch.softmax(final_output, dim=1)

        return probs.cpu().numpy()[0]