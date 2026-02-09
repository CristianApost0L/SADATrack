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
from src.constants import COCO_SWAP_PAIRS

def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# COCO skeleton connections for visualization (indices)
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
    """
    draw_joints = modality in ['joint', 'ensemble']
    draw_bones = modality in ['bone', 'ensemble']
    
    if draw_bones:
        for i, j in SKELETON_CONNECTIONS:
            if i < len(kpts) and j < len(kpts):
                if confs[i] > 0.25 and confs[j] > 0.25:
                    pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                    pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                    cv2.line(frame, pt1, pt2, color, 2)
            
    if draw_joints:
        for i in range(len(kpts)):
            if confs[i] > 0.25:
                c = (0, 0, 255) if draw_bones else (0, 255, 0)
                cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 4, c, -1)



def flip_keypoints_horizontal(kpts):
    """
    Flip keypoints horizontally to convert back-view to front-view convention.
    This swaps left/right indices to match anatomical labels.
    
    Args:
        kpts: (17, 3) array with [x, y, conf]
    Returns:
        Flipped keypoints with left/right swapped
    """
    kpts_flipped = kpts.copy()
    # Swap left/right pairs (anatomical correction)
    for left, right in COCO_SWAP_PAIRS:
        kpts_flipped[left] = kpts[right]
        kpts_flipped[right] = kpts[left]
    return kpts_flipped

def convert_to_bone(tensor_data):
    """Convert joint data to bone vectors."""
    bone_data = torch.zeros_like(tensor_data)
    for child, parent in COCO_BONE_PAIRS:
        bone_data[:, :2, :, child] = tensor_data[:, :2, :, child] - tensor_data[:, :2, :, parent]
        if tensor_data.size(1) > 2:
            bone_data[:, 2:, :, child] = tensor_data[:, 2:, :, child]
    return bone_data

def init_model(model_type, weights_path, num_classes, in_channels, device):
    print(f"Loading {model_type} from {weights_path}...")
    
    if not os.path.exists(weights_path):
         raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Auto-detect model type from checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    checkpoint_keys = checkpoint.keys()
    has_hdgcn_keys = any('conv_down' in key or 'aha.' in key for key in checkpoint_keys)
    has_ctrgcn_keys = any('alpha' in key or 'convs.' in key for key in checkpoint_keys)
    
    if has_hdgcn_keys and not has_ctrgcn_keys:
        detected_type = 'HDGCN'
    elif has_ctrgcn_keys and not has_hdgcn_keys:
        detected_type = 'CTRGCN'
    else:
        detected_type = model_type
    
    if detected_type != model_type:
        print(f"WARNING: Config specifies '{model_type}' but checkpoint is '{detected_type}'")
        print(f"Using detected type: {detected_type}")
        model_type = detected_type
    
    # Map 'HDGCN' -> HDGCN_Tennis, 'CTRGCN' -> CTRGCN_Tennis
    ModelClass = HDGCN_Tennis if model_type == 'HDGCN' else CTRGCN_Tennis
    
    model = ModelClass(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def main(args):
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model_hyperparameters']
    
    # Confidence threshold for swing detection
    CONFIDENCE_THRESHOLD = args.confidence
    BACK_VIEW = args.back_view  # Flag for players facing away from camera
    
    DATA_PROCESSED_DIR = data_config['processed_dir']
    
    local_processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    if os.path.exists(local_processed_dir) and os.path.exists(os.path.join(local_processed_dir, 'label_map.npy')):
        print(f"[INFO] Found locally processed data in {local_processed_dir}.")
        DATA_PROCESSED_DIR = local_processed_dir

    POSE_MODEL_PATH = config['model']['pose_extractor_path']
    SMART_CROP = config['model'].get('smart_crop', False)
    SEQ_LEN = config['hyperparameters'].get('seq_len', 40)
    IN_CHANNELS = model_config['in_channels']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    label_map = load_label_map(DATA_PROCESSED_DIR)
    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    model_type = model_config.get('type', 'HDGCN')
    
    weights_paths = args.weights.split(',')
    modality = args.modality.lower()
    
    classifier_joint = None
    classifier_bone = None
    
    if modality == 'ensemble':
        classifier_joint = init_model(model_type, weights_paths[0].strip(), num_classes, IN_CHANNELS, DEVICE)
        classifier_bone = init_model(model_type, weights_paths[1].strip(), num_classes, IN_CHANNELS, DEVICE)
    elif modality == 'joint':
        classifier_joint = init_model(model_type, weights_paths[0].strip(), num_classes, IN_CHANNELS, DEVICE)
    elif modality == 'bone':
        classifier_bone = init_model(model_type, weights_paths[0].strip(), num_classes, IN_CHANNELS, DEVICE)

    print(f"Loading Pose Estimator (YOLO-Pose)...")
    pose_model_path = args.pose_model if args.pose_model else POSE_MODEL_PATH
    extractor = PoseExtractor(
        model_path=pose_model_path,
        smart_crop=SMART_CROP
    )

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
    out = cv2.VideoWriter(output_path, fourcc, fps, (width , height))
    
    # State Management (2D only)
    # Target effective FPS ~30
    target_fps = 30
    stride = max(1, int(round(fps / target_fps)))
    mode_str = "2D Mode" + (" | Back View Correction Enabled" if BACK_VIEW else "")
    print(f"Video FPS: {fps}, Target FPS: {target_fps}, Stride: {stride}  [{mode_str}]")

    # Single player tracking
    pose_buffer = deque(maxlen=SEQ_LEN)
    pred_buffer = deque(maxlen=10)
    
    # --- PERSISTENT STATE ---
    # Holds the state to draw on SKIPPED frames
    last_detected_pose = None  # (norm, raw, confs, box)
    last_label = "Waiting..."
    last_conf = 0.0
    
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # === UPDATE LOGIC (Only on Stride) ===
        if frame_idx % stride == 0:
            
            # A. Detection - Use YOLO-Pose directly
            results = extractor.model(frame, verbose=False, conf=extractor.CONFIDENCE_THRESH)
            
            current_frame_pose = None
            
            if results and len(results[0].boxes) > 0:
                # Select the largest person (foreground player)
                boxes = results[0].boxes
                areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]
                best_idx = torch.argmax(areas).item()
                
                box = boxes[best_idx]
                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = xyxy
                
                # Smart Crop for better resolution
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
                    
                    # Re-run pose estimation on cropped region
                    pose_results = extractor.model(process_frame, verbose=False, conf=0.1)
                    
                    if pose_results and len(pose_results[0].boxes) > 0:
                        crop_areas = pose_results[0].boxes.xywh[:, 2] * pose_results[0].boxes.xywh[:, 3]
                        crop_best = torch.argmax(crop_areas).item()
                        kpts_raw = pose_results[0].keypoints.data[crop_best].cpu().numpy()
                        p_box_h = pose_results[0].boxes.xywh[crop_best, 3].item()
                        p_center = pose_results[0].boxes.xywh[crop_best, :2].cpu().numpy()
                        
                        # Remap to original frame coordinates
                        kpts_raw[:, 0] += offset_x
                        kpts_raw[:, 1] += offset_y
                        p_center[0] += offset_x
                        p_center[1] += offset_y
                    else:
                        # Fallback to initial detection
                        kpts_raw = results[0].keypoints.data[best_idx].cpu().numpy()
                        p_box_h = boxes.xywh[best_idx, 3].item()
                        p_center = boxes.xywh[best_idx, :2].cpu().numpy()
                else:
                    # No smart crop
                    kpts_raw = results[0].keypoints.data[best_idx].cpu().numpy()
                    p_box_h = boxes.xywh[best_idx, 3].item()
                    p_center = boxes.xywh[best_idx, :2].cpu().numpy()
                
                # Normalize keypoints
                norm_kpts = extractor._normalize(kpts_raw, p_box_h, p_center)
                
                # Apply back-view correction if enabled
                if BACK_VIEW:
                    norm_kpts = flip_keypoints_horizontal(norm_kpts)
                
                current_frame_pose = (norm_kpts, kpts_raw, kpts_raw[:, 2], xyxy)
            
            # Store in History Buffer
            if current_frame_pose is not None:
                norm, raw, confs, box = current_frame_pose
                pose_buffer.append(norm)
                last_detected_pose = current_frame_pose
            else:
                # Missing detection -> Append zeros to maintain temporal continuity
                pose_buffer.append(np.zeros((17, 3), dtype=np.float32))
                last_detected_pose = None 

            # B. Classification
            if len(pose_buffer) == SEQ_LEN:
                input_seq = np.array(pose_buffer, dtype=np.float32)
                non_zero_frames = np.sum(np.sum(input_seq, axis=(1, 2)) > 0)
                
                if non_zero_frames > SEQ_LEN // 2:
                    
                    # Normalize
                    input_seq_transposed = input_seq.transpose(2, 0, 1)  # (C, T, V)
                    input_seq_normalized = normalize_skeleton(input_seq_transposed)
                    
                    input_tensor = torch.tensor(input_seq_normalized).unsqueeze(0).to(DEVICE)
                    
                    probs = None
                    with torch.no_grad():
                        if classifier_joint:
                            probs = F.softmax(classifier_joint(input_tensor), dim=1)
                        if classifier_bone:
                            probs_b = F.softmax(classifier_bone(convert_to_bone(input_tensor)), dim=1)
                            probs = probs_b if probs is None else (probs + probs_b) / 2.0
                    
                    if probs is not None:
                        pred_buffer.append(probs.cpu().numpy())
                        avg_probs = np.mean(np.array(pred_buffer), axis=0).flatten()
                        top_loc = np.argmax(avg_probs)
                        top_conf = avg_probs[top_loc]
                        
                        # Only update label if confidence is above threshold
                        if top_conf >= CONFIDENCE_THRESHOLD:
                            last_label = idx_to_label[top_loc]
                            last_conf = top_conf
                        else:
                            # Low confidence - likely transition or no clear swing
                            last_label = "Uncertain"
                            last_conf = top_conf

        # === VISUALIZATION (Runs Every Frame) ===
        # Uses last_* variables for persistent drawing
        if last_detected_pose is not None:
            norm, raw, confs, box = last_detected_pose
            
            # Color based on prediction confidence
            if last_conf >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green: High confidence
            else:
                color = (0, 165, 255)  # Orange: Low confidence
            
            draw_skeleton(frame, raw[:, :2], confs, modality=modality, color=color)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            
            label_text = f"{last_label} ({last_conf:.2f})"
            cv2.putText(frame, label_text, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Frame: {frame_idx} | Buff: {len(pose_buffer)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', '--source', dest='source', required=True, help='Path to video file')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--modality', default='joint', choices=['joint', 'bone', 'ensemble'])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--pose_model', default=None)
    parser.add_argument('--output', default='demo_results/')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for swing detection (0.0-1.0, default: 0.5)')
    parser.add_argument('--back_view', action='store_true', help='Enable if player is facing away from camera (flips left/right for anatomical correctness)')
    
    args = parser.parse_args()
    main(args)
