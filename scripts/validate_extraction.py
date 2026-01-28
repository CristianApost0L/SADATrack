import os
import sys
import cv2
import argparse
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Aggiungi la directory del progetto al Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_thetis_files
from src.extractor import PoseExtractor
from ultralytics import YOLO

def load_config(config_path):
    """Carica il file YAML di configurazione"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# COCO skeleton connections (17 keypoints)
# Used to draw lines between keypoints
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs and Hips
    (5, 11), (6, 12), (5, 6),                         # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),                  # Arms
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)    # Head
]

def visualize_validation(video_path, extractor):
    """
    Performs frame-by-frame extraction and visualizes the result.
    Does not use _post_process (padding) to maintain synchronization with the original video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    frames_rgb = []
    frames_norm = []
    frames_raw = []

    print(f"Processing: {os.path.basename(video_path)}...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Convert BGR -> RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
        
        # --- EXTRACTION (Logic replicated from Extractor for debugging) ---
        # Using the extractor's model
        results = extractor.model(frame, verbose=False, conf=0.25)
        
        norm_kpts = np.zeros((17, 3)) # Default empty
        raw_kpts = np.zeros((17, 3))

        if results and len(results[0].boxes) > 0:
            # 1. Find main player (Max Area)
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            best_idx = torch.argmax(areas).item()
            
            # Raw data
            raw_kpts = results[0].keypoints.data[best_idx].cpu().numpy()
            box_h = results[0].boxes.xywh[best_idx, 3].item()
            box_center = results[0].boxes.xywh[best_idx, :2].cpu().numpy()

            # 2. Normalization (Call protected method for testing)
            norm_kpts = extractor._normalize(raw_kpts, box_h, box_center)

        frames_norm.append(norm_kpts)
        frames_raw.append(raw_kpts)

    cap.release()
    
    if len(frames_rgb) == 0:
        print("Empty or unreadable video.")
        return

    # --- ANIMATION SETUP ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Validation: {os.path.basename(video_path)}", fontsize=14)

    # Left plot: Original Video
    ax1.set_title("Input Video + YOLO Raw")
    im_display = ax1.imshow(frames_rgb[0])
    lines_raw = [ax1.plot([], [], 'c-', linewidth=2)[0] for _ in SKELETON_CONNECTIONS]
    points_raw = ax1.scatter([], [], c='r', s=10)

    # Right plot: Normalized Space
    ax2.set_title("Normalized Input (To LSTM)")
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0) # We invert Y in the plot because in CV Y grows downward
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(0, color='black', lw=1)
    
    lines_norm = [ax2.plot([], [], 'b-', linewidth=2)[0] for _ in SKELETON_CONNECTIONS]
    points_norm = ax2.scatter([], [], c='r', s=20)
    
    # Testo info
    info_text = ax2.text(-0.9, 0.9, "", fontsize=9)

    def update(frame_idx):
        # 1. Update Video
        im_display.set_data(frames_rgb[frame_idx])
        
        # Current data
        r_kpts = frames_raw[frame_idx]
        n_kpts = frames_norm[frame_idx]
        
        # --- Update Raw Skeleton (on Video) ---
        x_raw, y_raw, conf_raw = r_kpts[:, 0], r_kpts[:, 1], r_kpts[:, 2]
        
        # Update lines
        for line, (i, j) in zip(lines_raw, SKELETON_CONNECTIONS):
            if conf_raw[i] > 0.1 and conf_raw[j] > 0.1:
                line.set_data([x_raw[i], x_raw[j]], [y_raw[i], y_raw[j]])
            else:
                line.set_data([], [])
        # Update points
        mask_r = conf_raw > 0.1
        points_raw.set_offsets(np.c_[x_raw[mask_r], y_raw[mask_r]])

        # --- Update Normalized Skeleton (su Grafico) ---
        # NOTA: Invertiamo la Y (-y) per visualizzarlo "in piedi" nel grafico cartesiano
        # (perché nei video la Y cresce verso il basso, nei grafici verso l'alto)
        x_norm, y_norm, conf_norm = n_kpts[:, 0], n_kpts[:, 1], n_kpts[:, 2]
        
        for line, (i, j) in zip(lines_norm, SKELETON_CONNECTIONS):
            if conf_norm[i] > 0.1 and conf_norm[j] > 0.1:
                line.set_data([x_norm[i], x_norm[j]], [-y_norm[i], -y_norm[j]])
            else:
                line.set_data([], [])
                
        mask_n = conf_norm > 0.1
        points_norm.set_offsets(np.c_[x_norm[mask_n], -y_norm[mask_n]])
        
        # Check centering (Bacino)
        hip_x = (x_norm[11] + x_norm[12]) / 2
        info_text.set_text(f"Frame: {frame_idx}\nHip Center X: {hip_x:.3f}")

        return [im_display, points_raw, points_norm] + lines_raw + lines_norm

    ani = FuncAnimation(fig, update, frames=len(frames_rgb), interval=50, blit=True)
    plt.show()

def main(config_path):
    # Carica la configurazione
    config = load_config(config_path)
    DATA_RAW_DIR = config['data']['raw_dir']
    MODEL_YOLO_PATH = config['model']['yolo_path']
    SEQ_LEN = config['hyperparameters']['seq_len']
    NUM_JOINTS = config['hyperparameters']['num_joints']
    CONFIDENCE_THRESH = config['hyperparameters']['confidence_thresh']
    
    # 1. Initialize extractor
    print("Loading model for validation...")
    try:
        extractor = PoseExtractor(
            model_path=MODEL_YOLO_PATH,
            seq_len=SEQ_LEN,
            num_joints=NUM_JOINTS,
            confidence_thresh=CONFIDENCE_THRESH
        )
    except Exception as e:
        print(f"Error: {e}")
        # Fallback local path if config path does not exist
        extractor = PoseExtractor(model_path='yolov8x-pose.pt')

    # 2. Get video list
    try:
        samples, _ = get_thetis_files(DATA_RAW_DIR)
    except FileNotFoundError:
        print("Dataset not found. Check DATA_RAW_DIR in src/config.py")
        return

    if not samples:
        print("No videos found.")
        return

    # 3. Interactive loop
    while True:
        # Choose a random video
        target_video, label_id = random.choice(samples)
        
        print(f"\n--- VALIDATION ---")
        print(f"Video: {target_video}")
        print(f"Class ID: {label_id}")
        
        visualize_validation(target_video, extractor)
        
        # Ask to continue
        ans = input("\nPress ENTER for another video, or 'q' to exit: ")
        if ans.lower() == 'q':
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate pose extraction from videos')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()
    
    main(args.config)