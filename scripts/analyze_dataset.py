import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
        print(f"     Recommendations:")
        print(f"     • Use weighted metrics: F1-Score, Precision, Recall")
        print(f"     • Consider class weights in loss function")
        print(f"     • Use stratified cross-validation (already in place ✓)")
    else:
        print(f"\n OK Classes are well balanced")
    print(f"{'='*60}\n")
    
    # Create visualization
    create_distribution_plot(label_map, counts, ANALYSIS_OUTPUT_DIR)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze dataset class distribution')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()
    
    analyze_dataset(args.config)
