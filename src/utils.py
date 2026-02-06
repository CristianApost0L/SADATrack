"""
Utility functions for file handling and dataset management.
"""
import os
from .constants import THETIS_CLASSES, LABEL_MAP


def _normalize_string(s):
    """Normalize string for class name matching."""
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def get_thetis_files(root_dir):
    """
    Get list of video files from TheTis dataset directory structure.
    
    Args:
        root_dir: Root directory containing class subdirectories
    
    Returns:
        samples: List of (video_path, class_id) tuples
        LABEL_MAP: Dictionary mapping class names to indices
    """
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
