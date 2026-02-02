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

def main(config_path, reuse_2d=False):

    config = load_config(config_path)
    DATA_RAW_DIR = config['data']['raw_dir']
    DATA_PROCESSED_DIR = config['data']['processed_dir']
    MODEL_YOLO_PATH = config['model']['yolo_path']
    SEQ_LEN = config['hyperparameters']['seq_len']
    NUM_JOINTS = config['hyperparameters']['num_joints']
    CONFIDENCE_THRESH = config['hyperparameters']['confidence_thresh']
    
    # 1. Setup Directory
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # --- REUSE EXISTING 2D DATA LOGIC ---
    if reuse_2d:
        X_path = os.path.join(DATA_PROCESSED_DIR, 'X.npy')
        if os.path.exists(X_path):
            print(f"\n[INFO] Reusing existing 2D data from {X_path}")
            X_old = np.load(X_path)
            
            # Formato atteso (N, T, V, C) per iterazione
            if X_old.ndim == 4 and X_old.shape[1] < 5: # Probabile (N, C, T, V)
                 print(f"Detected (N, C, T, V) format: {X_old.shape}. Transposing...")
                 X_old = X_old.transpose(0, 2, 3, 1)
            
            print(f"Loaded data shape: {X_old.shape}")
            
            use_motionbert = config['model'].get('use_motionbert', False)
            if use_motionbert:
                from src.motionbert_extractor import MotionBERTExtractor
                mb_path = config['model'].get('motionbert_path', '')
                try:
                    lifter = MotionBERTExtractor(checkpoint_path=mb_path)
                except Exception as e:
                    print(f"Lifter Init Error: {e}")
                    lifter = MotionBERTExtractor(checkpoint_path=mb_path)

                if lifter.valid:
                    print("Lifting 2D Keypoints to 3D...")
                    X_new = []
                    for i in tqdm(range(len(X_old)), desc="Lifting Sequences"):
                        sample_2d = X_old[i]
                        kpts_3d = lifter.lift_2d_to_3d(sample_2d)
                        
                        if sample_2d.shape[-1] >= 3:
                            conf = sample_2d[:, :, 2:3]
                            sample_final = np.concatenate([kpts_3d, conf], axis=2)
                        else:
                            sample_final = kpts_3d
                        X_new.append(sample_final)
                    
                    X = np.array(X_new, dtype=np.float32)
                    print(f"New 3D Data Shape: {X.shape}")
                    
                    if os.path.exists(X_path) and os.access(os.path.dirname(X_path), os.W_OK):
                         os.rename(X_path, X_path.replace('.npy', '_2d_backup.npy'))
                    elif not os.access(os.path.dirname(X_path), os.W_OK):
                         print(f"[WARNING] Input directory is read-only. Saving to local working directory instead.")
                         local_output_dir = os.path.join(os.getcwd(), 'data', 'processed')
                         os.makedirs(local_output_dir, exist_ok=True)
                         X_path = os.path.join(local_output_dir, 'X.npy')
                    
                    # Converti per salvare nel formato del dataset (N, C, T, V) ? 
                    # Dataset.py si aspetta (N, C, T, V) in __init__ ma fa permute se serve?
                    # prepare_data di solito salva (N, T, V, C) e src/dataset.py in __init__ fa permute(0, 3, 1, 2)
                    
                    print(f"Saving 3D data to {X_path}...")
                    np.save(X_path, X)
                    
                    # Ensure y.npy and label_map.npy are also in the destination folder
                    y_src = os.path.join(DATA_PROCESSED_DIR, 'y.npy')
                    label_map_src = os.path.join(DATA_PROCESSED_DIR, 'label_map.npy')
                    
                    y_dst = os.path.join(os.path.dirname(X_path), 'y.npy')
                    label_map_dst = os.path.join(os.path.dirname(X_path), 'label_map.npy')
                    
                    if os.path.exists(y_src) and y_src != y_dst:
                        print(f"Copying y.npy to {y_dst}...")
                        import shutil
                        shutil.copy2(y_src, y_dst)
                        
                    if os.path.exists(label_map_src) and label_map_src != label_map_dst:
                        print(f"Copying label_map.npy to {label_map_dst}...")
                        import shutil
                        shutil.copy2(label_map_src, label_map_dst)
                        
                    print("Done. Ready for training.")
                    return

    # 2. Extractor Initialization
    print(f"Initialization Pose Extractor...")
    
    use_motionbert = config['model'].get('use_motionbert', False)
    
    try:
        if use_motionbert:
            print(" Using MotionBERT for 3D Lifting...")
            from src.motionbert_extractor import MotionBERTIntegratedExtractor
            mb_path = config['model'].get('motionbert_path', '')
            extractor = MotionBERTIntegratedExtractor(
                yolo_path=MODEL_YOLO_PATH,
                motionbert_ckpt=mb_path
            )
        else:
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
    parser.add_argument('--reuse_2d', action='store_true', 
                        help='Skip video extraction and reuse existing 2D X.npy for 3D lifting')
    args = parser.parse_args()
    
    main(args.config, reuse_2d=args.reuse_2d)