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

from src.model import HDGCN_Tennis, CTRGCN_Tennis
from src.extractor import PoseExtractor
from src.dataset import COCO_BONE_PAIRS, normalize_skeleton
try:
    from src.motionbert_extractor import MotionBERTExtractor
except ImportError:
    MotionBERTExtractor = None
    print("Warning: Could not import MotionBERTExtractor. 3D lifting will not be available.")

def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# COCO skeleton connections for visualization (indices)
# Used for drawing lines
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

def draw_skeleton(frame, kpts, confs, modality='joint', color=(0, 255, 0)):
    """
    Draw skeleton on frame with pixel coordinates.
    modality: 'joint', 'bone', 'ensemble'
    """
    draw_joints = modality in ['joint', 'ensemble']
    draw_bones = modality in ['bone', 'ensemble']
    
    # Draw Bones (Lines)
    if draw_bones:
        for i, j in SKELETON_CONNECTIONS:
            if confs[i] > 0.25 and confs[j] > 0.25:
                pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                cv2.line(frame, pt1, pt2, color, 2)
            
    # Draw Joints (Circles)
    if draw_joints:
        for i in range(17):
            if confs[i] > 0.25:
                # Different color for joints vs bones
                c = (0, 0, 255) if draw_bones else (0, 255, 0)
                cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 4, c, -1)

def convert_to_bone(tensor_data):
    """
    Convert joint data to bone vectors.
    tensor_data: (N, C, T, V)
    """
    bone_data = torch.zeros_like(tensor_data)
    for child, parent in COCO_BONE_PAIRS:
        # Vector = Child - Parent (for X, Y)
        bone_data[:, :2, :, child] = tensor_data[:, :2, :, child] - tensor_data[:, :2, :, parent]
        # Keep confidence if exists
        if tensor_data.size(1) > 2:
            bone_data[:, 2, :, child] = tensor_data[:, 2, :, child]
    return bone_data

def init_model(model_type, weights_path, num_classes, in_channels, device):
    print(f"Loading {model_type} from {weights_path}...")
    if model_type == 'HDGCN':
        model = HDGCN_Tennis(num_classes=num_classes, in_channels=in_channels)
    elif model_type == 'CTRGCN':
        model = CTRGCN_Tennis(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if not os.path.exists(weights_path):
         raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main(args):
    # Load configuration
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model_hyperparameters']
    
    DATA_PROCESSED_DIR = data_config['processed_dir']
    
    # Check for locally processed data (priority over config)
    local_processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    if os.path.exists(local_processed_dir) and os.path.exists(os.path.join(local_processed_dir, 'label_map.npy')):
        print(f"[INFO] Found locally processed data in {local_processed_dir}. Using this instead of config path.")
        DATA_PROCESSED_DIR = local_processed_dir

    MODEL_YOLO_PATH = config['model']['yolo_path']
    SEQ_LEN = config['hyperparameters'].get('seq_len', 60)
    USE_MOTIONBERT = config['model'].get('use_motionbert', False)
    MOTIONBERT_PATH = config['model'].get('motionbert_path', '')
    IN_CHANNELS = model_config['in_channels']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup
    label_map = load_label_map(DATA_PROCESSED_DIR)
    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # Determine model type from config (architecture)
    model_type = model_config.get('type', 'HDGCN')
    
    # Parse Weights and Modality
    weights_paths = args.weights.split(',')
    modality = args.modality.lower()
    
    classifier_joint = None
    classifier_bone = None
    
    if modality == 'ensemble':
        if len(weights_paths) != 2:
            print("ERROR: For ensemble mode, provide two weights files separated by comma (joint_weights.pth,bone_weights.pth)")
            return
        classifier_joint = init_model(model_type, weights_paths[0].strip(), num_classes, IN_CHANNELS, DEVICE)
        classifier_bone = init_model(model_type, weights_paths[1].strip(), num_classes, IN_CHANNELS, DEVICE)
    elif modality == 'joint':
        classifier_joint = init_model(model_type, weights_paths[0].strip(), num_classes, IN_CHANNELS, DEVICE)
    elif modality == 'bone':
        classifier_bone = init_model(model_type, weights_paths[0].strip(), num_classes, IN_CHANNELS, DEVICE)
    else:
        print(f"ERROR: Unknown modality {modality}")
        return

    print(f"Loading Pose Estimator (YOLO)...")
    pose_model_path = args.pose_model if args.pose_model else MODEL_YOLO_PATH
    extractor = PoseExtractor(model_path=pose_model_path)

    # Initialize MotionBERT Lifter if required
    lifter = None
    if USE_MOTIONBERT:
        if MotionBERTExtractor is None:
             print("ERROR: MotionBERT is required by config but could not be imported.")
             return
        print(f"Loading MotionBERT Lifter from {MOTIONBERT_PATH}...")
        lifter = MotionBERTExtractor(checkpoint_path=MOTIONBERT_PATH, device=DEVICE)

    # 2. Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {args.source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith(('/', '\\')):
        output_path = os.path.join(output_path, 'demo_output.mp4')
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    sequence_buffer = deque(maxlen=SEQ_LEN)
    prediction_buffer = deque(maxlen=5) 
    
    current_label = "Waiting for sequence..."
    current_conf = 0.0

    print(f"Processing video: {args.source}")
    print(f"Mode: {modality.upper()}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"3D Lifting: {'ENABLED' if lifter else 'DISABLED'}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. Pose Estimation ---
        results = extractor.model(frame, verbose=False, conf=0.25)
        
        norm_kpts = np.zeros((17, 3), dtype=np.float32)
        raw_kpts = None
        has_detection = False
        
        if results and len(results[0].boxes) > 0:
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            best_idx = torch.argmax(areas).item()
            raw_kpts = results[0].keypoints.data[best_idx].cpu().numpy() # x, y, conf
            box_h = results[0].boxes.xywh[best_idx, 3].item()
            box_center = results[0].boxes.xywh[best_idx, :2].cpu().numpy()
            
            # Normalization
            norm_kpts = extractor._normalize(raw_kpts, box_h, box_center)
            has_detection = True
            
            # Custom Visualization
            # Use a COPY of frame to avoid cumulative drawing if debugging
            draw_skeleton(frame, raw_kpts[:, :2], raw_kpts[:, 2], modality=modality)

        sequence_buffer.append(norm_kpts)
        
        # --- B. Classification ---
        # Add visual indicator for buffer status
        buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQ_LEN}"
        cv2.putText(frame, buffer_status, (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if len(sequence_buffer) == SEQ_LEN:
            input_seq = np.array(sequence_buffer, dtype=np.float32)
            
            # Check if sequence has enough valid frames (not just zeros)
            non_zero_frames = np.sum(np.sum(input_seq, axis=(1, 2)) > 0)
            
            if non_zero_frames > SEQ_LEN // 2:
                
                # --- 3D Lifting (if enabled) ---
                if lifter:
                    # Input to lifter: (T, V, C) or (T, V, 2)
                    kpts_3d = lifter.lift_2d_to_3d(input_seq)
                    
                    # Merge with original confidence (index 2)
                    if input_seq.shape[-1] >= 3:
                        conf = input_seq[:, :, 2:3]
                    else:
                        conf = np.ones((input_seq.shape[0], input_seq.shape[1], 1), dtype=np.float32)
                        
                    # Final shape: (T, V, 4) -> x, y, z, conf
                    input_seq = np.concatenate([kpts_3d, conf], axis=2)

                # --- 4. Dataset Normalization (CRITICAL) ---
                # The model was trained on data normalized by `normalize_skeleton`.
                # We must apply the same transform here.
                # normalize_skeleton handles frame-by-frame invariance (centering, rotation, scaling).
                
                # Input: (T, V, C) -> Transpose to (C, T, V)
                input_seq_transposed = input_seq.transpose(2, 0, 1)
                
                # Apply normalization
                input_seq_normalized = normalize_skeleton(input_seq_transposed)
                
                # Create tensor: (1, C, T, V)
                input_tensor = torch.tensor(input_seq_normalized).unsqueeze(0).to(DEVICE)
                
                probs = None
                
                with torch.no_grad():
                    # 1. Joint Branch
                    if classifier_joint is not None:
                        logits_joint = classifier_joint(input_tensor)
                        probs_joint = F.softmax(logits_joint, dim=1)
                        probs = probs_joint
                    
                    # 2. Bone Branch
                    if classifier_bone is not None:
                        bone_tensor = convert_to_bone(input_tensor)
                        logits_bone = classifier_bone(bone_tensor)
                        probs_bone = F.softmax(logits_bone, dim=1)
                        probs = probs_bone
                    
                    # 3. Ensemble
                    if classifier_joint is not None and classifier_bone is not None:
                        # Average fusion
                        probs = (probs_joint + probs_bone) / 2.0

                # DEBUG info
                max_p = torch.max(probs).item()
                if max_p > 0.99 or np.isnan(max_p):
                     pass

                prediction_buffer.append(probs.cpu().numpy())
                avg_probs = np.mean(prediction_buffer, axis=0)
                pred_idx = np.argmax(avg_probs)
                current_conf = avg_probs[0, pred_idx]
                
                if current_conf > 0.5:
                    current_label = idx_to_label[pred_idx]
                else:
                    current_label = "Uncertain"
            else:
                 # Not enough detection in buffer
                 current_conf = 0.0
                 current_label = "No Person Detected"

        cv2.rectangle(frame, (0, 0), (450, 90), (0, 0, 0), -1)
        
        color = (0, 255, 0) if current_conf > 0.75 else (0, 255, 255)
        
        cv2.putText(frame, f"Action: {current_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {current_conf:.2f} ({modality})", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '--source', dest='source', required=True, help='Path to video file')
    parser.add_argument('--weights', required=True, help='Path to model weights. For ensemble, separate with comma: "joint.pth,bone.pth"')
    parser.add_argument('--modality', default='joint', choices=['joint', 'bone', 'ensemble'], help='Input modality for model')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--pose_model', default=None, help='Path to YOLO pose model')
    parser.add_argument('--output', default='demo_results/', help='Output directory or file')
    
    args = parser.parse_args()
    main(args)
