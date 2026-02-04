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

# H36M skeleton connections for 3D visualization (MotionBERT format)
H36M_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),      # Right Leg
    (0, 4), (4, 5), (5, 6),      # Left Leg
    (0, 7), (7, 8),              # Spine/Torso
    (8, 9), (9, 10),             # Neck/Head
    (8, 11), (11, 12), (12, 13), # Left Arm
    (8, 14), (14, 15), (15, 16)  # Right Arm
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
            # Check range to avoid index errors
            if i < len(kpts) and j < len(kpts):
                if confs[i] > 0.25 and confs[j] > 0.25:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    cv2.line(frame, pt1, pt2, color, 2)
            
    # Draw Joints (Circles)
    if draw_joints:
        for i in range(len(kpts)):
            if confs[i] > 0.25:
                # Different color for joints vs bones
                c = (0, 0, 255) if draw_bones else (0, 255, 0)
                cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 4, c, -1)

def draw_3d_skeleton_on_canvas(canvas, kpts_3d, offset_x=150, offset_y=150, scale=120, color=(0, 255, 255)):
    """
    Draws 3D skeleton on a separate canvas.
    Displays two views: Front (X, Y) and Side (Z, Y).
    """
    # kpts_3d shape: (17, 3) -> (X, Y, Z)
    # Assuming MotionBERT output: Y is vertical, Z is depth.
    # We might need to handle specific coordinate systems. 
    # Usually: Y is Down (image coordinates) or Up.
    
    # 1. Front View (X, Y) - Left Half of Canvas
    # 2. Side View (Z, Y) - Right Half of Canvas
    
    # Centering
    # Find root (Hip center)
    root = kpts_3d[0] # In H36M, index 0 is Pelvis (Root)
    kpts_centered = kpts_3d - root
    
    # Define offsets for the two views on the canvas
    # Canvas assumed to be roughly 300-400px wide
    
    view_1_offset = (offset_x // 2, offset_y)
    view_2_offset = (offset_x + (offset_x // 2), offset_y)
    
    # Use H36M connections for 3D data
    for i, j in H36M_SKELETON_CONNECTIONS:
        if i >= len(kpts_centered) or j >= len(kpts_centered): continue
        
        p1 = kpts_centered[i]
        p2 = kpts_centered[j]
        
        # --- VIEW 1: Front (X, Y) ---

        # Note: In 3D pose, often Y is DOWN. If it looks upside down, flip Y sign.
        u1 = int(p1[0] * scale + view_1_offset[0])
        v1 = int(p1[1] * scale + view_1_offset[1])
        u2 = int(p2[0] * scale + view_1_offset[0])
        v2 = int(p2[1] * scale + view_1_offset[1])
        cv2.line(canvas, (u1, v1), (u2, v2), color, 2)
        cv2.circle(canvas, (u1, v1), 3, (255, 255, 0), -1) 
        
        # --- VIEW 2: Side (Z, Y) ---
        # Using Z as the horizontal axis
        u1_s = int(p1[2] * scale + view_2_offset[0]) 
        v1_s = int(p1[1] * scale + view_2_offset[1])
        u2_s = int(p2[2] * scale + view_2_offset[0]) 
        v2_s = int(p2[1] * scale + view_2_offset[1])
        cv2.line(canvas, (u1_s, v1_s), (u2_s, v2_s), (0, 0, 255), 2)
        cv2.circle(canvas, (u1_s, v1_s), 3, (255, 0, 255), -1)

    # Labels
    cv2.putText(canvas, "Front (X-Y)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Side (Z-Y)", (offset_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



def convert_to_bone(tensor_data):
    """
    Convert joint data to bone vectors.
    tensor_data: (N, C, T, V)
    """
    bone_data = torch.zeros_like(tensor_data)
    for child, parent in COCO_BONE_PAIRS:
        # Vector = Child - Parent (for X, Y)
        bone_data[:, :2, :, child] = tensor_data[:, :2, :, child] - tensor_data[:, :2, :, parent]
        # Keep Z and Confidence (start from index 2)
        if tensor_data.size(1) > 2:
            bone_data[:, 2:, :, child] = tensor_data[:, 2:, :, child]
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
    COURTSIDE_PATH = config['model'].get('courtside_path', None)
    DETECTOR_PATH = config['model'].get('detector_path', None)
    SMART_CROP = config['model'].get('smart_crop', False)
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
    extractor = PoseExtractor(
        model_path=pose_model_path, 
        courtside_path=COURTSIDE_PATH,
        detector_path=DETECTOR_PATH,
        smart_crop=SMART_CROP
    )

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
    
    # Extra width for 3D visualization box (e.g., 400px wide)
    vis_3d_width = 400 if lifter else 0
    out = cv2.VideoWriter(output_path, fourcc, fps, (width + vis_3d_width, height))
    
    sequence_buffer = deque(maxlen=SEQ_LEN)
    prediction_buffer = deque(maxlen=5) 
    
    current_label = "Waiting for sequence..."
    current_conf = 0.0

    print(f"Processing video: {args.source}")
    print(f"Mode: {modality.upper()}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"3D Lifting: {'ENABLED' if lifter else 'DISABLED'}")
    
    # --- MULTI-PLAYER SETUP ---
    NUM_PLAYERS = 2
    # Buffers for each player (0: Top/Far, 1: Bottom/Near)
    # Note: _select_targets_with_racket sorts by Y coordinate (Top first)
    player_buffers = [deque(maxlen=SEQ_LEN) for _ in range(NUM_PLAYERS)]
    player_preds_buffer = [deque(maxlen=5) for _ in range(NUM_PLAYERS)]
    p_labels = ["Waiting...", "Waiting..."]
    p_confs = [0.0, 0.0]
    
    # Last seen 3D skeletons for visualization persistence
    last_3d_skeletons = [None, None] 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Prepare 3D Canvas
        if lifter:
            canvas_3d = np.zeros((height, vis_3d_width, 3), dtype=np.uint8)
            cv2.line(canvas_3d, (vis_3d_width//2, 0), (vis_3d_width//2, height), (50, 50, 50), 1)

        
        # --- A. Detection & Pose Extraction ---
        # 1. Detect Persons
        det_results = extractor.detector(frame, verbose=False, classes=[0], conf=extractor.CONFIDENCE_THRESH)
        
        # 2. Detect Rackets (for sorting/filtering)
        racket_boxes = None
        if extractor.has_racket_detector:
            try:
                r_res = extractor.racket_detector(frame, verbose=False, conf=0.15)
                if r_res and len(r_res[0].boxes) > 0:
                    racket_boxes = r_res[0].boxes
                    
                    # Draw Rackets (Blue) for Debugging
                    for box in racket_boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "Racket", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
            except Exception:
                pass # Ignore racket failure
        
        
        person_boxes = det_results[0].boxes if det_results and len(det_results[0].boxes) > 0 else None
        
        # Identify Targets
        target_indices = []
        if person_boxes:
            target_indices = extractor._select_targets_with_racket(person_boxes, racket_boxes, num_targets=NUM_PLAYERS)
        
        # Process each target
        current_frame_poses = [] # List of (norm_kpts, raw_kpts, confs, box_xywh)
        
        for i, box_idx in enumerate(target_indices):
            # Box Info
            box = person_boxes[box_idx]
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = xyxy
            
            # --- Smart Crop Logic (Replicated from extractor) ---
            process_frame = frame
            offset_x, offset_y = 0, 0
            
            if extractor.smart_crop:
                h_img, w_img = frame.shape[:2]
                box_w, box_h = x2 - x1, y2 - y1
                pad_w, pad_h = box_w * 0.2, box_h * 0.2
                
                crop_x1 = max(0, int(x1 - pad_w))
                crop_y1 = max(0, int(y1 - pad_h))
                crop_x2 = min(w_img, int(x2 + pad_w))
                crop_y2 = min(h_img, int(y2 + pad_h))
                
                process_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                offset_x, offset_y = crop_x1, crop_y1
            
            # Pose Inference on crop/target
            # Use lower conf for crop
            pose_results = extractor.model(process_frame, verbose=False, conf=0.1)
            
            if pose_results and len(pose_results[0].boxes) > 0:
                # Find main person in crop
                areas = pose_results[0].boxes.xywh[:, 2] * pose_results[0].boxes.xywh[:, 3]
                best_idx = torch.argmax(areas).item()
                
                kpts_raw = pose_results[0].keypoints.data[best_idx].cpu().numpy() # (17, 3)
                p_box_h = pose_results[0].boxes.xywh[best_idx, 3].item()
                p_center = pose_results[0].boxes.xywh[best_idx, :2].cpu().numpy()
                
                # Remap to global
                kpts_raw[:, 0] += offset_x
                kpts_raw[:, 1] += offset_y
                p_center[0] += offset_x
                p_center[1] += offset_y
                
                # Normalize
                norm_kpts = extractor._normalize(kpts_raw, p_box_h, p_center)
                
                current_frame_poses.append((norm_kpts, kpts_raw, kpts_raw[:, 2], box.xyxy.cpu().numpy()[0]))
            else:
                current_frame_poses.append(None) # Detected person but no pose found
        
        # --- B. Update Buffers & Draw ---
        for p_idx in range(NUM_PLAYERS):
            # If we have a detected pose for this player index
            if p_idx < len(current_frame_poses) and current_frame_poses[p_idx] is not None:
                norm, raw, confs, box = current_frame_poses[p_idx]
                player_buffers[p_idx].append(norm)
                
                # Draw Skeleton
                # Color code: P0 (Top) = Red-ish, P1 (Bottom) = Green-ish
                color = (0, 0, 255) if p_idx == 0 else (0, 255, 0)
                draw_skeleton(frame, raw[:, :2], confs, modality=modality, color=color)
                
                # Draw Box
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                
                # Draw Label
                label_text = f"P{p_idx}: {p_labels[p_idx]} ({p_confs[p_idx]:.2f})"
                cv2.putText(frame, label_text, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            else:
                # Missing player in this frame
                if len(player_buffers[p_idx]) > 0: # Repeat last if exists, or push zero? 
                    # Generally zeroes or repeat. Let's push zeroes to indicate missing data.
                    player_buffers[p_idx].append(np.zeros((17, 3), dtype=np.float32))
                else:
                     player_buffers[p_idx].append(np.zeros((17, 3), dtype=np.float32))

        # --- C. Classification (Per Player) ---
        for p_idx in range(NUM_PLAYERS):
            p_buff = player_buffers[p_idx]
            
            if len(p_buff) == SEQ_LEN:
                input_seq = np.array(p_buff, dtype=np.float32)
                non_zero_frames = np.sum(np.sum(input_seq, axis=(1, 2)) > 0)
                
                if non_zero_frames > SEQ_LEN // 2:
                    # Lifting
                    if lifter:
                        kpts_3d_seq = lifter.lift_2d_to_3d(input_seq)
                        # Save LAST frame 3D skeleton for drawing
                        last_3d_skeletons[p_idx] = kpts_3d_seq[-1]
                        
                        conf = input_seq[:, :, 2:3] if input_seq.shape[-1] >= 3 else np.ones((SEQ_LEN, 17, 1))
                        input_seq = np.concatenate([kpts_3d_seq, conf], axis=2)
                    
                    # Normalize & Prepare

                    input_seq_transposed = input_seq.transpose(2, 0, 1) # (T, V, C) -> (C, T, V)
                    input_seq_normalized = normalize_skeleton(input_seq_transposed)
                    input_tensor = torch.tensor(input_seq_normalized).unsqueeze(0).to(DEVICE)
                    
                    probs = None
                    with torch.no_grad():
                        if classifier_joint:
                             probs = F.softmax(classifier_joint(input_tensor), dim=1)
                        if classifier_bone:
                             probs_b = F.softmax(classifier_bone(convert_to_bone(input_tensor)), dim=1)
                             probs = probs_b if probs is None else (probs + probs_b)/2.0
                    
                    # Store & Averaging
                    player_preds_buffer[p_idx].append(probs.cpu().numpy())
                    
                    # Smooth prediction
                    avg_probs = np.mean(np.array(player_preds_buffer[p_idx]), axis=0).flatten()
                    top_loc = np.argmax(avg_probs)
                    p_labels[p_idx] = idx_to_label[top_loc]
                    p_confs[p_idx] = avg_probs[top_loc]

        # Global Status
        cv2.putText(frame, f"Frame: {len(player_buffers[0])}/{SEQ_LEN}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Display Current Prediction
        y_pos = 60
        for p_idx in range(NUM_PLAYERS):
            
            # --- Draw 3D if available ---
            if lifter and last_3d_skeletons[p_idx] is not None:
                # Vertical stacking for 2 players
                # Top half for P0, Bottom half for P1
                center_y = int(height * 0.25) if p_idx == 0 else int(height * 0.75)
                draw_3d_skeleton_on_canvas(canvas_3d, last_3d_skeletons[p_idx], 
                                          offset_x=vis_3d_width//2, 
                                          offset_y=center_y, 
                                          scale=100, 
                                          color=color)

        # Write to video
        final_frame = frame
        if lifter:
             final_frame = np.hstack((frame, canvas_3d))
             
        out.write(final_frame)
        
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
