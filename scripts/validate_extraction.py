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

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalization import normalize_skeleton
from src.dataset import get_thetis_files, THETIS_CLASSES
from src.extractor import PoseExtractor
from src.constants import SKELETON_CONNECTIONS, COCO_CONNECTIONS
from ultralytics import YOLO

def load_config(config_path):
    """Carica il file YAML di configurazione"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_validation(video_path, extractor, confidence_thresh=0.25, output_dir='runs/validation'):
    """
    Performs frame-by-frame extraction and saves visualization video.

    Args:
        video_path: Path to the video file
        extractor: PoseExtractor instance
        confidence_thresh: Confidence threshold for Pose Estimation
        output_dir: Directory to save output video
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
        
        # Extraction
        results = extractor.model(frame, verbose=False, conf=confidence_thresh)
        
        norm_kpts = np.zeros((17, 3)) # Default empty (will be filled by normalize_skeleton)
        raw_kpts = np.zeros((17, 3))

        if results and len(results[0].boxes) > 0:
            # 1. Find main player (Max Area)
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            best_idx = torch.argmax(areas).item()
            
            # Raw data
            raw_kpts = results[0].keypoints.data[best_idx].cpu().numpy()

        frames_norm.append(norm_kpts)  # Placeholder, will be replaced by normalization
        frames_raw.append(raw_kpts)

    cap.release()
    
    if len(frames_rgb) == 0:
        print("Empty or unreadable video.")
        return
    
    print(f"✓ Extracted {len(frames_rgb)} frames")
    
    # Apply normalization to the entire sequence
    print("Applying normalization ...")
    
    # Convert list to array (T, V, C)
    raw_sequence = np.array(frames_raw, dtype=np.float32)
    
    # Apply normalize_skeleton: (T, V, C) -> (C, T, V) -> normalize -> (T, V, C)
    if raw_sequence.shape[0] > 0:
        seq_transposed = raw_sequence.transpose(2, 0, 1)  # (C, T, V)
        seq_normalized = normalize_skeleton(seq_transposed)  # (C, T, V)
        frames_norm = seq_normalized.transpose(1, 2, 0)  # (T, V, C)
        frames_norm = [frames_norm[i] for i in range(len(frames_norm))]  # Convert back to list
        print(f"✓ Normalization applied: torso-based scaling, centered on hips")
    
    # Animation setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Validation: {os.path.basename(video_path)} ({len(frames_rgb)} frames)", fontsize=14)

    # Left: Raw keypoints on video
    ax1.set_title("Raw Keypoints (Video Frame)", fontsize=12)
    ax1.axis('off')
    im_display = ax1.imshow(frames_rgb[0])
    lines_raw = [ax1.plot([], [], 'c-', linewidth=2)[0] for _ in SKELETON_CONNECTIONS]
    points_raw = ax1.scatter([], [], c='r', s=10)

    # Right: Normalized skeleton
    ax2.set_title("Robust Normalization (Torso-based)", fontsize=12)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y (flipped)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(0, color='black', lw=1)
    
    
    lines_norm = [ax2.plot([], [], 'b-', linewidth=2)[0] for _ in SKELETON_CONNECTIONS]
    points_norm = ax2.scatter([], [], c='r', s=20)
    info_text = ax2.text(-2.2, 2.2, "", fontsize=9)

    def update(frame_idx):
        # Update Video
        im_display.set_data(frames_rgb[frame_idx])
        
        # Current data
        r_kpts = frames_raw[frame_idx]
        n_kpts = frames_norm[frame_idx]
        
        # Update Raw Skeleton (on Video)
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

        # Update Normalized Skeleton
        x_norm, y_norm, conf_norm = n_kpts[:, 0], n_kpts[:, 1], n_kpts[:, 2]
        
        for line, (i, j) in zip(lines_norm, SKELETON_CONNECTIONS):
            if conf_norm[i] > 0.1 and conf_norm[j] > 0.1:
                line.set_data([x_norm[i], x_norm[j]], [-y_norm[i], -y_norm[j]])
            else:
                line.set_data([], [])
                
        mask_n = conf_norm > 0.1
        points_norm.set_offsets(np.c_[x_norm[mask_n], -y_norm[mask_n]])
        
        # Check centering and scaling (from normalization)
        hip_x = (x_norm[11] + x_norm[12]) / 2 if conf_norm[11] > 0.1 and conf_norm[12] > 0.1 else 0
        hip_y = (y_norm[11] + y_norm[12]) / 2 if conf_norm[11] > 0.1 and conf_norm[12] > 0.1 else 0
        
        # Torso length (shoulder to hip center)
        if conf_norm[5] > 0.1 and conf_norm[6] > 0.1:
            shoulder_x = (x_norm[5] + x_norm[6]) / 2
            shoulder_y = (y_norm[5] + y_norm[6]) / 2
            torso_len = np.sqrt((shoulder_x - hip_x)**2 + (shoulder_y - hip_y)**2)
            info_text.set_text(f"Frame: {frame_idx}\n"
                             f"Hip Center: ({hip_x:.3f}, {hip_y:.3f})\n"
                             f"Torso Length: {torso_len:.3f}")
        else:
            info_text.set_text(f"Frame: {frame_idx}\n"
                             f"Hip Center: ({hip_x:.3f}, {hip_y:.3f})")
        
        return [im_display, points_raw, points_norm] + lines_raw + lines_norm

    ani = FuncAnimation(fig, update, frames=len(frames_rgb), interval=50, blit=True)
    
    # Save video
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"validation_{video_name}.mp4")
    
    print(f"Saving validation video to: {output_path}")
    print("This may take a few moments...")
    
    try:
        # Save as MP4 using ffmpeg writer
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Tennis Swing Classifier'), bitrate=1800)
        ani.save(output_path, writer=writer)
        print(f"✓ Video saved successfully!")
    except Exception as e:
        print(f"⚠ FFMpeg not available, trying Pillow writer...")
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=20)
            output_path_gif = output_path.replace('.mp4', '.gif')
            ani.save(output_path_gif, writer=writer)
            print(f"✓ GIF saved to: {output_path_gif}")
        except Exception as e2:
            print(f"✗ Error saving animation: {e2}")
            print("Install ffmpeg or pillow to save videos.")
    
    plt.close(fig)

def debug_prepared_data(data_path='data/processed/X.npy', output_dir='runs/debug'):
    """
    Debug prepared data to check if the format is COCO and show the normalization.
    
    Args:
        data_path: Path to the prepared X.npy file
        output_dir: Directory to save debug visualizations
    """
    if not os.path.exists(data_path):
        print(f"File not found at {data_path}.")
        print("Please run 'python scripts/prepare_data.py' first to generate the data.")
        
        # Try alternative paths
        local_path = os.path.join(os.getcwd(), 'data', 'processed', 'X.npy')
        if os.path.exists(local_path):
            data_path = local_path
            print(f"Found data at {local_path}. Using this path instead.")
        else:
            print("No data found. Please run 'python scripts/prepare_data.py' first.")
            return 

    print("\n=== PREPARED DATA DEBUG ===")
    print(f"Loading data from: {data_path}")
    X = np.load(data_path)
    print(f"Data shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Try to load labels too
    y_path = data_path.replace('X.npy', 'y.npy')
    if os.path.exists(y_path):
        y = np.load(y_path)
        print(f"Labels shape: {y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {np.bincount(y)}")
    
    sample = X[0]  # Take the first sample
    print(f"\nSample shape: {sample.shape}")
    
    # Detect format and extract first frame
    if sample.shape[0] in [2, 3, 4]:  # (C, T, V) format
        print("Detected format: (C, T, V)")
        C, T, V = sample.shape
        frame = sample[:, 0, :]  # Take the first frame
        x = frame[0, :]
        y_coord = frame[1, :]
        print(f"Channels={C}, Time={T}, Vertices={V}")
    elif sample.shape[-1] in [2, 3, 4]:  # (T, V, C) format
        print("Detected format: (T, V, C)")
        T, V, C = sample.shape
        frame = sample[0, :, :]  # Take the first frame
        x = frame[:, 0]
        y_coord = frame[:, 1]
        print(f"Time={T}, Vertices={V}, Channels={C}")
    else:
        print(f"⚠ Unexpected data shape: {sample.shape}. Cannot determine format.")
        return

    print(f"\nFirst frame keypoints:")
    print(f"  X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Y range: [{y_coord.min():.3f}, {y_coord.max():.3f}]")
    print(f"  Num keypoints: {len(x)}")
     
    # Visualization (2D only - COCO format)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Prepared Data Debug: {os.path.basename(data_path)}", fontsize=14)
    
    # Plot 1: COCO Connections
    ax1 = axes[0]
    ax1.set_title("COCO Format (17 joints)")
    ax1.scatter(x, -y_coord, c='blue', s=50, zorder=3)
    for i, j in COCO_CONNECTIONS:
        if i < len(x) and j < len(x):
            ax1.plot([x[i], x[j]], [-y_coord[i], -y_coord[j]], 'b-', linewidth=2, alpha=0.6)
    # Add joint numbers
    for i in range(len(x)):
        ax1.text(x[i], -y_coord[i], str(i), fontsize=8, ha='center', va='bottom')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('-Y (inverted)')
    
    # Plot 2: Standard SKELETON_CONNECTIONS (used in training)
    ax2 = axes[1]
    ax2.set_title("Training Format (SKELETON_CONNECTIONS)")
    ax2.scatter(x, -y_coord, c='green', s=50, zorder=3)
    for i, j in SKELETON_CONNECTIONS:
        if i < len(x) and j < len(x):
            ax2.plot([x[i], x[j]], [-y_coord[i], -y_coord[j]], 'g-', linewidth=2, alpha=0.6)
    # Add joint numbers
    for i in range(len(x)):
        ax2.text(x[i], -y_coord[i], str(i), fontsize=8, ha='center', va='bottom')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('-Y (inverted)')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'debug_skeleton_format.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Skeleton visualization saved to: {output_path}")
    
    # Show interpretation
    print("\n=== FORMAT INTERPRETATION ===")
    print("If the skeleton looks correct in:")
    print("  - LEFT plot  → Data is in COCO format ✓")
    print("  - RIGHT plot → Training will use this topology")
    print("\nNote: All data should be in COCO format (17 joints)")
    print("\nPress any key to close the plot...")
    plt.show()

def main(config_path, mode='video', output_dir=None, num_videos=None, process_all=False, video_path=None):
    """
    Main debug function.
    
    Args:
        config_path: Path to config.yaml
        mode: 'video' for video extraction validation, 'data' for prepared data validation
        output_dir: Directory to save validation videos (overrides config if provided)
        num_videos: Number of random videos to process (default: None = interactive)
        process_all: Process all videos in dataset (default: False)
        video_path: Process a specific video file (default: None)
    """
    if mode == 'data':
        # Debug prepared data
        config = load_config(config_path)
        DATA_PROCESSED_DIR = config['data'].get('processed_dir', 'data/processed')
        X_PATH = os.path.join(DATA_PROCESSED_DIR, 'X.npy')
        OUTPUT_DIR = config['output'].get('analysis_output_dir', 'runs/analysis')
        debug_prepared_data(X_PATH, OUTPUT_DIR)
        return
    
    config = load_config(config_path)
    DATA_RAW_DIR = config['data']['raw_dir']
    MODEL_YOLO_PATH = config['model']['pose_extractor_path']
    SEQ_LEN = config['hyperparameters']['seq_len']
    NUM_JOINTS = config['hyperparameters']['num_joints']
    CONFIDENCE_THRESH = config['hyperparameters']['confidence_thresh']
    
    # Read validation output directory from config (can be overridden by parameter)
    if output_dir is None:
        output_dir = config['output'].get('validation_output_dir', 'runs/validation')
    
    # 1. Initialize extractor
    print("Loading model for validation...")
    print(f"  - Model: {MODEL_YOLO_PATH}")
    print(f"  - Sequence length: {SEQ_LEN}")
    print(f"  - Num joints: {NUM_JOINTS}")
    print(f"  - Confidence threshold: {CONFIDENCE_THRESH}")
    print(f"  - Visualization: 2D only")
    
    if not os.path.exists(MODEL_YOLO_PATH):
        print(f"\n⚠ Warning: Model path '{MODEL_YOLO_PATH}' not found!")
        print("  Falling back to default 'yolov8x-pose.pt'")
        MODEL_YOLO_PATH = 'yolov8x-pose.pt'
    
    extractor = PoseExtractor(
        model_path=MODEL_YOLO_PATH,
        seq_len=SEQ_LEN,
        num_joints=NUM_JOINTS,
        confidence_thresh=CONFIDENCE_THRESH
    )

    # 2. Get video list
    try:
        samples, _ = get_thetis_files(DATA_RAW_DIR)
    except FileNotFoundError:
        print(f"Dataset not found. Check raw_dir in config: {DATA_RAW_DIR}")
        return

    if not samples:
        print("No videos found.")
        return

    print(f"\nFound {len(samples)} videos in dataset")

    # 3. Determine processing mode
    if video_path:
        # Single video mode
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return
        
        print(f"\n--- PROCESSING SINGLE VIDEO ---")
        print(f"Video: {video_path}")
        visualize_validation(video_path, extractor, CONFIDENCE_THRESH, output_dir)
        print("\n✓ Done!")
        return
    
    elif process_all:
        # Process all videos mode (non-interactive)
        print(f"\n--- PROCESSING ALL {len(samples)} VIDEOS ---")
        for i, (target_video, label_id) in enumerate(samples, 1):
            class_name = THETIS_CLASSES[label_id] if 0 <= label_id < len(THETIS_CLASSES) else "Unknown"
            print(f"\n[{i}/{len(samples)}] {class_name}: {os.path.basename(target_video)}")
            visualize_validation(target_video, extractor, CONFIDENCE_THRESH, output_dir)
        print("\n✓ All videos processed!")
        return
    
    elif num_videos:
        # Process N random videos mode (non-interactive)
        num_to_process = min(num_videos, len(samples))
        selected = random.sample(samples, num_to_process)
        print(f"\n--- PROCESSING {num_to_process} RANDOM VIDEOS ---")
        
        for i, (target_video, label_id) in enumerate(selected, 1):
            class_name = THETIS_CLASSES[label_id] if 0 <= label_id < len(THETIS_CLASSES) else "Unknown"
            print(f"\n[{i}/{num_to_process}] {class_name}: {os.path.basename(target_video)}")
            visualize_validation(target_video, extractor, CONFIDENCE_THRESH, output_dir)
        print("\n✓ All videos processed!")
        return

    # Interactive loop (default, local use only)
    print("\n--- INTERACTIVE MODE ---")
    print("(Not recommended for Kaggle/Colab - use --num-videos or --all instead)\n")
    while True:
        # Choose a random video
        target_video, label_id = random.choice(samples)
        class_name = THETIS_CLASSES[label_id] if 0 <= label_id < len(THETIS_CLASSES) else "Unknown"
        
        print(f"\n--- VALIDATION ---")
        print(f"Video: {target_video}")
        print(f"Class: {class_name} (ID: {label_id})")
        
        visualize_validation(target_video, extractor, CONFIDENCE_THRESH, output_dir)
        
        # Ask to continue
        ans = input("\nProcess another video? (y/N): ")
        if ans.lower() != 'y':
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Debug and validate pose extraction pipeline',
        epilog='Examples:\n'
               '  Interactive (local):     python validate_extraction.py\n'
               '  Process 5 videos:        python validate_extraction.py --num-videos 5\n'
               '  Process all:             python validate_extraction.py --all\n'
               '  Specific video:          python validate_extraction.py --video path/to/video.mp4\n'
               '  Kaggle/Colab batch mode: python validate_extraction.py --num-videos 10 --output /kaggle/working',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'data'],
                        help='Debug mode: "video" for raw video extraction validation, "data" for prepared data validation (default: video)')
    parser.add_argument('--output', type=str, default='runs/validation',
                        help='Output directory for validation videos (default: runs/validation)')
    parser.add_argument('--num-videos', type=int, default=None,
                        help='Process N random videos (non-interactive, good for Kaggle/Colab)')
    parser.add_argument('--all', action='store_true',
                        help='Process all videos in dataset (non-interactive)')
    parser.add_argument('--video', type=str, default=None,
                        help='Process a specific video file')
    args = parser.parse_args()
    
    main(args.config, args.mode, args.output, args.num_videos, args.all, args.video)