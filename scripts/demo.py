import os
import sys
import cv2
import torch
import yaml
import argparse
import numpy as np
from collections import deque
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HDGCN_Tennis
from src.extractor import PoseExtractor

def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# COCO skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs
    (5, 11), (6, 12), (5, 6),                         # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),                  # Arms
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)    # Head
]

def load_label_map(data_processed_dir):
    path = os.path.join(data_processed_dir, 'label_map.npy')
    if not os.path.exists(path):
        print(f"ERROR: Label map not found at {path}. Run prepare_data.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=True).item()

def draw_skeleton(frame, kpts, confs, color=(0, 255, 0)):
    """Draw skeleton on frame with pixel coordinates."""
    for i, j in SKELETON_CONNECTIONS:
        if confs[i] > 0.25 and confs[j] > 0.25:
            pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(frame, pt1, pt2, color, 2)
            
    for i in range(17):
        if confs[i] > 0.25:
            cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 4, (0, 0, 255), -1)

def main(args):
    # Load configuration
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model_hyperparameters']
    
    DATA_PROCESSED_DIR = data_config['processed_dir']
    MODEL_YOLO_PATH = config['model']['yolo_path']
    IN_CHANNELS = model_config['in_channels']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup
    label_map = load_label_map(DATA_PROCESSED_DIR)
    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    print(f"Loading Classifier model: {args.weights}")
    classifier = HDGCN_Tennis(num_classes=num_classes, in_channels=3)
    classifier.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    classifier.to(DEVICE)
    classifier.eval()

    print(f"Loading Pose Estimator (YOLO)...")
    # Using extractor for consistency
    pose_model_path = args.pose_model if args.pose_model else MODEL_YOLO_PATH
    extractor = PoseExtractor(model_path=pose_model_path)

    # 2. Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {args.source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle output path: if it's a directory, add default filename
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith(('/', '\\')):
        output_path = os.path.join(output_path, 'demo_output.mp4')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 3. Sliding window buffer
    # Network expects 40 frames. Using deque with max 40 elements.
    sequence_buffer = deque(maxlen=40)
    
    # Smoothing prediction with moving average on probabilities
    prediction_buffer = deque(maxlen=5) 
    
    current_label = "Waiting for sequence..."
    current_conf = 0.0

    print(f"Processing video: {args.source}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. Pose Estimation ---
        results = extractor.model(frame, verbose=False, conf=0.25)
        
        # Data for this frame
        norm_kpts = np.zeros((17, 3), dtype=np.float32) # Normalized for network
        raw_kpts = None # Pixel coordinates for visualization
        
        if results and len(results[0].boxes) > 0:
            # Select player with max bounding box area
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            best_idx = torch.argmax(areas).item()
            
            # Extract raw keypoint data
            raw_kpts = results[0].keypoints.data[best_idx].cpu().numpy() # x, y, conf
            box_h = results[0].boxes.xywh[best_idx, 3].item()
            box_center = results[0].boxes.xywh[best_idx, :2].cpu().numpy()
            
            # NORMALIZATION (identical to training!)
            # Using PoseExtractor._normalize method
            norm_kpts = extractor._normalize(raw_kpts, box_h, box_center)
            
            # Visualization
            draw_skeleton(frame, raw_kpts[:, :2], raw_kpts[:, 2])

        # Add to buffer (even if empty/zero, to maintain correct timing)
        sequence_buffer.append(norm_kpts)
        
        # --- B. Classification ---
        # Only when buffer is filled with 40 frames
        if len(sequence_buffer) == 40:
            # Prepare tensor: (1, 3, 40, 17)
            # sequence_buffer is (40, 17, 3) -> stack -> numpy
            input_seq = np.array(sequence_buffer, dtype=np.float32)
            
            # Check if sequence is valid (not all zeros)
            if np.sum(input_seq) > 0:
                input_tensor = torch.tensor(input_seq).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    
                # Add to smoothing buffer
                prediction_buffer.append(probs.cpu().numpy())
                
                # Average probabilities over last 5 frames (stability)
                avg_probs = np.mean(prediction_buffer, axis=0)
                pred_idx = np.argmax(avg_probs)
                current_conf = avg_probs[0, pred_idx]
                
                if current_conf > 0.5: # Visualization threshold
                    current_label = idx_to_label[pred_idx]
                else:
                    current_label = "Uncertain"

        # --- C. Draw Interface ---
        # Black background for text
        cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
        
        # Color: Green if high confidence, Yellow if medium
        color = (0, 255, 0) if current_conf > 0.75 else (0, 255, 255)
        
        cv2.putText(frame, f"Action: {current_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {current_conf:.2f}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo: Swing classification on video with real-time inference')
    parser.add_argument('--source', type=str, required=True, help='Path to input video')
    parser.add_argument('--weights', type=str, default='models/best_model_hdgcn.pth', help='Path to trained model (.pth)')
    parser.add_argument('--output', type=str, default='demo_output.mp4', help='Path to output video')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML file (default: config.yaml)')
    parser.add_argument('--pose_model', type=str, default=None, help='Path to YOLO pose model (optional, uses config.yaml if not provided)')
    
    args = parser.parse_args()
    main(args)