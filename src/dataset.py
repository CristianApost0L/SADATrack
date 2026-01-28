import os
import numpy as np
import torch
from torch.utils.data import Dataset

# The 12 official TheTis classes
THETIS_CLASSES = [
    "backhand2hands",
    "backhand", 
    "backhand_slice",
    "backhand_volley",
    "forehand_flat",
    "forehand_openstands",
    "forehand_slice",
    "forehand_volley",
    "flat_service",
    "kick_service",
    "slice_service",
    "smash"
]

# Mapping from class name to label index
LABEL_MAP = {cls_name: i for i, cls_name in enumerate(THETIS_CLASSES)}

def _normalize_string(s):
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

def get_thetis_files(root_dir):

    samples = []
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    normalized_classes = {_normalize_string(k): k for k in THETIS_CLASSES}
    
    try:
        subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    except NotADirectoryError:
        print(f"ERROR: {root_dir} does not appear to be a directory.")
        return [], LABEL_MAP

    mapped_folders = 0
    
    for folder_name in subfolders:
        clean_folder = _normalize_string(folder_name)
        official_name = None
        
        if clean_folder in normalized_classes:
            official_name = normalized_classes[clean_folder]
        else:
            for norm_cls, real_cls in normalized_classes.items():
                if norm_cls in clean_folder:
                    official_name = real_cls
                    break
        
        if official_name:
            mapped_folders += 1
            class_id = LABEL_MAP[official_name]
            folder_path = os.path.join(root_dir, folder_name)
            
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.avi', '.mp4', '.mov'))]
            
            for video_file in files:
                full_path = os.path.join(folder_path, video_file)
                samples.append((full_path, class_id))
                
            print(f"  [OK] Directory '{folder_name}' mapped with '{official_name}' with {len(files)} videos.")
        else:
            print(f"  [SKIP] Directory '{folder_name}' could not be mapped to any official class.")

    return samples, LABEL_MAP

def augment_skeleton(data, rotation_range=10, scale_range=0.1, noise_std=0.005):
    """
    Applies random rotation, scaling and jittering.
    input data shape: (C, T, V) -> (Channels, Time, Vertices)
    """
    C, T, V = data.shape
    
    # Separate (X, Y) from Confidence
    if C >= 2:
        data_xy = data[:2, :, :]   
        data_rest = data[2:, :, :] 
    else:
        return data 

    # 1. Rotation (around Z axis / camera view)
    theta = np.radians(np.random.uniform(-rotation_range, rotation_range))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]]) 
    
    # Reshape for fast matrix multiplication: (2, T*V)
    xy_reshaped = data_xy.reshape(2, -1)
    xy_rotated = np.dot(rotation_matrix, xy_reshaped)
    data_xy = xy_rotated.reshape(2, T, V)
    
    # 2. Scaling (Simulated zoom)
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    data_xy = data_xy * scale
    
    # 3. Gaussian Noise (Sensor jittering)
    noise_xy = np.random.normal(0, noise_std, data_xy.shape)
    data_xy = data_xy + noise_xy
    
    # Reconstruction
    if C > 2:
        data_final = np.concatenate([data_xy, data_rest], axis=0)
    else:
        data_final = data_xy
    
    return data_final.astype(np.float32)


class TennisDataset(Dataset):
    def __init__(self, X, y, augment=False):
        """
        Args:
            X: numpy array (N, T, V, C) from prepare_data.py
            y: numpy array (N,) labels
            augment: bool, if True applies random transformations
        """
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2) 
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx].clone().numpy() 
        label = self.y[idx]
        
        if self.augment:
            sample = augment_skeleton(sample)
            
        return torch.FloatTensor(sample), label