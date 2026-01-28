import os
import sys
import argparse
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractor import PoseExtractor
from src.dataset import get_thetis_files

def load_config(config_path):
    """Carica il file YAML di configurazione"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):

    config = load_config(config_path)
    DATA_RAW_DIR = config['data']['raw_dir']
    DATA_PROCESSED_DIR = config['data']['processed_dir']
    MODEL_YOLO_PATH = config['model']['yolo_path']
    SEQ_LEN = config['hyperparameters']['seq_len']
    NUM_JOINTS = config['hyperparameters']['num_joints']
    CONFIDENCE_THRESH = config['hyperparameters']['confidence_thresh']
    
    # 1. Setup Directory
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # 2. Yolo Model Initialization
    print(f"Inizialization Pose Extractor...")
    try:
        extractor = PoseExtractor(
            model_path=MODEL_YOLO_PATH,
            seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
            confidence_thresh=CONFIDENCE_THRESH
        )
    except Exception as e:
        print(f"Error during model loading: {e}")
        return

    # 3. Scan Dataset
    try:
        samples, label_map = get_thetis_files(DATA_RAW_DIR)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Number of video samples found: {len(samples)}")
    
    if len(samples) == 0:
        print("No video files found in the specified directory.")
        print(f"Please check the path: {DATA_RAW_DIR}")
        return

    # 4. Extract Features
    X_data = []
    y_data = []
    
    print("\nKeypoint Extraction in progress...")
    for video_path, label in tqdm(samples, desc="Extracting keypoints", unit="video"):
        kpts = extractor.extract_sequence(video_path)
        
        if kpts is not None:
            X_data.append(kpts)
            y_data.append(label)
            
    # 5. Saving Data
    if len(X_data) > 0:
        print(f"\nSaving processed data to {DATA_PROCESSED_DIR}...")
        
        # Progress bar for data conversion
        with tqdm(total=3, desc="Converting data", unit="step") as pbar:
            X = np.array(X_data, dtype=np.float32)
            pbar.update(1)
            y = np.array(y_data, dtype=np.int64)
            pbar.update(1)
            # Prepare label_map for saving
            pbar.update(1)
        
        # Progress bar for saving files
        with tqdm(total=3, desc="Saving files", unit="file") as pbar:
            np.save(os.path.join(DATA_PROCESSED_DIR, 'X.npy'), X)
            pbar.update(1)
            np.save(os.path.join(DATA_PROCESSED_DIR, 'y.npy'), y)
            pbar.update(1)
            np.save(os.path.join(DATA_PROCESSED_DIR, 'label_map.npy'), label_map)
            pbar.update(1)
        
        print(f"\nCOMPLETED: Data saved successfully in {DATA_PROCESSED_DIR}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Classes saved: {label_map}")
        print(f"\nRun 'python scripts/analyze_dataset.py' to analyze class distribution")
    else:
        print("ERROR: No keypoints were extracted from any video. Process terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset by extracting pose keypoints')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()
    
    main(args.config)