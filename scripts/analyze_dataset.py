import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_dataset(config_path):
    """Analyze dataset class distribution"""
    # Load configuration
    config = load_config(config_path)
    data_config = config['data']
    output_config = config['output']
    
    DATA_PROCESSED_DIR = data_config['processed_dir']

    # Check for locally processed data (priority over config)
    local_processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    if os.path.exists(local_processed_dir) and os.path.exists(os.path.join(local_processed_dir, 'X.npy')):
        print(f"[INFO] Found locally processed data in {local_processed_dir}. Using this instead of config path.")
        DATA_PROCESSED_DIR = local_processed_dir
    
    ANALYSIS_OUTPUT_DIR = output_config.get('analysis_output_dir', 'runs/analysis')
    
    # Load data
    X_path = os.path.join(DATA_PROCESSED_DIR, 'X.npy')
    y_path = os.path.join(DATA_PROCESSED_DIR, 'y.npy')
    label_map_path = os.path.join(DATA_PROCESSED_DIR, 'label_map.npy')
    
    if not os.path.exists(X_path):
        print(f"ERROR: Data not found at {DATA_PROCESSED_DIR}")
        print(f"Please run 'python scripts/prepare_data.py' first")
        return
    
    X = np.load(X_path)
    y = np.load(y_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()
    num_classes = len(label_map)
    
    # Print dataset statistics
    print(f"\n{'='*60}")
    print(f"Dataset Analysis")
    print(f"{'='*60}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Analyze class distribution
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    print(f"\nClass Distribution:")
    for class_idx, count in zip(unique, counts):
        class_idx = int(class_idx)  # Convert to standard Python int
        class_name = label_map.get(class_idx, f"Class {class_idx}")
        percentage = (count / total_samples) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"  {class_name:15s}: {count:4d} ({percentage:5.2f}%) {bar}")
    
    # Check for class imbalance
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\n{'='*60}")
    print(f"Class Balance Analysis:")
    print(f"  Max samples: {max_count}")
    print(f"  Min samples: {min_count}")
    print(f"  Balance Ratio (max/min): {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2:
        print(f"\n  WARNING: Classes are significantly imbalanced!")
    else:
        print(f"\n OK Classes are well balanced")
    print(f"{'='*60}\n")
    
    # Create visualization
    create_distribution_plot(label_map, counts, ANALYSIS_OUTPUT_DIR)
    
    # Analyze confidence values (if available)
    analyze_confidence(X, y, label_map, ANALYSIS_OUTPUT_DIR)

def create_distribution_plot(label_map, counts, output_dir='runs/analysis'):
    """Create and save class distribution plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Invert label_map: from {class_name: index} to {index: class_name}
    index_to_name = {v: k for k, v in label_map.items()}
    class_names = [index_to_name.get(i, f"Class {i}") for i in range(len(label_map))]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, counts, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    print(f"Class distribution plot saved to: {os.path.join(output_dir, 'class_distribution.png')}")
    plt.close()

def analyze_confidence(X, y, label_map, output_dir='runs/analysis'):
    """Analyze confidence values for each joint and swing class"""
    
    # COCO-17 joint names
    COCO_JOINT_NAMES = [
        'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
        'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
        'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip',
        'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
    ]
    
    print(f"\n{'='*60}")
    print(f"Confidence Analysis")
    print(f"{'='*60}")
    
    # Detect data format
    print(f"\nData shape: {X.shape}")
    
    # Handle different data formats: (N, T, V, C) or (N, C, T, V)
    if X.ndim == 4:
        # Check if channels are in position 1 or 3
        if X.shape[1] <= 4:  # (N, C, T, V) format
            print(f"Detected format: (N={X.shape[0]}, C={X.shape[1]}, T={X.shape[2]}, V={X.shape[3]})")
            # Transpose to (N, T, V, C)
            X_analysis = X.transpose(0, 2, 3, 1)
        else:  # (N, T, V, C) format
            print(f"Detected format: (N={X.shape[0]}, T={X.shape[1]}, V={X.shape[2]}, C={X.shape[3]})")
            X_analysis = X
    else:
        print(f"Unexpected data shape: {X.shape}")
        return
    
    N, T, V, C = X_analysis.shape
    
    # Check if confidence channel exists
    if C < 3:
        print(f"\nWARNING: No confidence channel found (C={C} < 3)")
        print(f"Confidence analysis requires at least 3 channels (X, Y, Confidence)")
        return
    
    # Extract confidence channel (3rd channel: X, Y, Conf format)
    confidences = X_analysis[:, :, :, 2]  # (N, T, V)
    
    print(f"\nConfidence statistics:")
    print(f"  Shape: {confidences.shape}")
    print(f"  Overall mean: {confidences.mean():.4f}")
    print(f"  Overall std:  {confidences.std():.4f}")
    print(f"  Min: {confidences.min():.4f}")
    print(f"  Max: {confidences.max():.4f}")
    
    # Prepare label mapping
    index_to_name = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # Calculate mean confidence per joint per class
    conf_per_class_joint = np.zeros((num_classes, V))
    
    for class_idx in range(num_classes):
        class_mask = (y == class_idx)
        if class_mask.sum() > 0:
            class_confidences = confidences[class_mask]  # (N_class, T, V)
            # Average over samples and time
            conf_per_class_joint[class_idx] = class_confidences.mean(axis=(0, 1))
    
    # Print per-class confidence summary
    print(f"\n{'='*60}")
    print(f"Mean Confidence per Swing Class:")
    print(f"{'='*60}")
    
    for class_idx in range(num_classes):
        class_name = index_to_name.get(class_idx, f"Class {class_idx}")
        mean_conf = conf_per_class_joint[class_idx].mean()
        std_conf = conf_per_class_joint[class_idx].std()
        print(f"  {class_name:20s}: {mean_conf:.4f} ± {std_conf:.4f}")
    
    # Print per-joint confidence summary (averaged across all classes)
    print(f"\n{'='*60}")
    print(f"Mean Confidence per Joint (all classes):")
    print(f"{'='*60}")
    
    overall_joint_conf = conf_per_class_joint.mean(axis=0)
    for joint_idx, joint_name in enumerate(COCO_JOINT_NAMES):
        if joint_idx < V:
            conf = overall_joint_conf[joint_idx]
            # Visual indicator
            bar_length = int(conf * 30)
            bar = '█' * bar_length
            print(f"  {joint_name:12s}: {conf:.4f} {bar}")
    
    # Identify problematic joints (low confidence)
    print(f"\n{'='*60}")
    print(f"Low-Confidence Joints (potential YOLO detection issues):")
    print(f"{'='*60}")
    
    threshold = 0.5
    low_conf_joints = [(i, COCO_JOINT_NAMES[i], overall_joint_conf[i]) 
                       for i in range(min(V, len(COCO_JOINT_NAMES))) 
                       if overall_joint_conf[i] < threshold]
    
    if low_conf_joints:
        for idx, name, conf in sorted(low_conf_joints, key=lambda x: x[2]):
            print(f"  {name:12s}: {conf:.4f} - May cause issues in real matches")
    else:
        print(f"  None (all joints > {threshold})")
    
    # Create visualizations
    create_confidence_heatmap(conf_per_class_joint, label_map, COCO_JOINT_NAMES, output_dir)
    create_confidence_barplot(overall_joint_conf, COCO_JOINT_NAMES, output_dir)
    create_confidence_boxplot(confidences, y, label_map, output_dir)
    
    print(f"\n{'='*60}\n")

def create_confidence_heatmap(conf_matrix, label_map, joint_names, output_dir='runs/analysis'):
    """Create heatmap of confidence per class and joint"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare labels
    index_to_name = {v: k for k, v in label_map.items()}
    class_names = [index_to_name.get(i, f"Class {i}") for i in range(len(label_map))]
    
    # Truncate joint names for better visualization
    V = conf_matrix.shape[1]
    joint_labels = joint_names[:V]
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(conf_matrix, 
                xticklabels=joint_labels,
                yticklabels=class_names,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Mean Confidence'})
    
    plt.title('Mean Confidence per Joint and Swing Class', fontsize=14, fontweight='bold')
    plt.xlabel('Joint', fontsize=12)
    plt.ylabel('Swing Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confidence_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confidence heatmap saved to: {output_path}")
    plt.close()

def create_confidence_barplot(overall_conf, joint_names, output_dir='runs/analysis'):
    """Create bar plot of overall confidence per joint"""
    os.makedirs(output_dir, exist_ok=True)
    
    V = len(overall_conf)
    joint_labels = joint_names[:V]
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if c < 0.5 else 'orange' if c < 0.7 else 'green' for c in overall_conf]
    bars = plt.bar(joint_labels, overall_conf, color=colors, alpha=0.7)
    
    # Add horizontal threshold lines
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Low (< 0.5)')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Medium (< 0.7)')
    
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, overall_conf)):
        plt.text(bar.get_x() + bar.get_width()/2., conf + 0.02,
                f'{conf:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Joint', fontsize=12)
    plt.ylabel('Mean Confidence', fontsize=12)
    plt.title('Overall Confidence per Joint (THETIS Dataset)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confidence_per_joint.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confidence bar plot saved to: {output_path}")
    plt.close()

def create_confidence_boxplot(confidences, y, label_map, output_dir='runs/analysis'):
    """Create box plot of confidence distribution per class"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    index_to_name = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    
    # Flatten confidences per class
    data_for_boxplot = []
    labels_for_boxplot = []
    
    for class_idx in range(num_classes):
        class_mask = (y == class_idx)
        if class_mask.sum() > 0:
            class_confidences = confidences[class_mask]
            # Flatten all confidences for this class (all samples, frames, joints)
            flattened = class_confidences.flatten()
            # Remove zeros (missing detections)
            flattened = flattened[flattened > 0]
            
            data_for_boxplot.append(flattened)
            labels_for_boxplot.append(index_to_name.get(class_idx, f"Class {class_idx}"))
    
    # Create box plot
    plt.figure(figsize=(14, 6))
    bp = plt.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    plt.xlabel('Swing Class', fontsize=12)
    plt.ylabel('Confidence Distribution', fontsize=12)
    plt.title('Confidence Distribution per Swing Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.05)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Low threshold')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='Medium threshold')
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confidence_distribution_per_class.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confidence distribution plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset class distribution')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()
    
    analyze_dataset(args.config)
