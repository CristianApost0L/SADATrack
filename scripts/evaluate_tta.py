import os
import sys
import yaml
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import TennisDataset
from src.model import HDGCN_Tennis, CTRGCN_Tennis


def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_processed_dir):
    """Load processed data and label map"""
    try:
        X = np.load(os.path.join(data_processed_dir, 'X.npy'))
        y = np.load(os.path.join(data_processed_dir, 'y.npy'))
        label_map = np.load(os.path.join(data_processed_dir, 'label_map.npy'), allow_pickle=True).item()
        return X, y, label_map
    except FileNotFoundError as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def apply_tta(model, inputs, device, tta_augmentations):
    """Apply Test Time Augmentation with multiple passes"""
    all_probs = []
    
    # Original pass
    with torch.no_grad():
        if hasattr(model, 'use_hyperbolic') and model.use_hyperbolic:
            output, _ = model(inputs, return_embeddings=True)
        else:
            output = model(inputs)
        prob = F.softmax(output, dim=1)
        all_probs.append(prob)
    
    # TTA augmentations
    for aug_config in tta_augmentations:
        scale = aug_config.get('scale', 1.0)
        noise = aug_config.get('noise', 0.0)
        
        inputs_aug = inputs.clone()
        
        # Apply scaling to X, Y coordinates (channels 0, 1)
        if scale != 1.0:
            inputs_aug[:, :2, :, :] *= scale
        
        # Apply noise to X, Y coordinates
        if noise > 0:
            noise_tensor = torch.randn_like(inputs_aug[:, :2, :, :]) * noise
            inputs_aug[:, :2, :, :] += noise_tensor
        
        with torch.no_grad():
            if hasattr(model, 'use_hyperbolic') and model.use_hyperbolic:
                output, _ = model(inputs_aug, return_embeddings=True)
            else:
                output = model(inputs_aug)
            prob = F.softmax(output, dim=1)
            all_probs.append(prob)
    
    # Ensemble: average all predictions
    avg_prob = torch.stack(all_probs).mean(dim=0)
    return avg_prob

def main(config_path):
    """Main evaluation function with TTA"""
    # Load configuration
    config = load_config(config_path)
    data_config = config['data']
    output_config = config['output']
    evaluation_config = config['evaluation']
    model_config = config['model_hyperparameters']
    training_config = config.get('training', {})
    MODALITY = training_config.get('modality', 'joint')

    DATA_PROCESSED_DIR = data_config['processed_dir']
    
    # Check for locally processed data (priority over config)
    local_processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    if os.path.exists(local_processed_dir) and os.path.exists(os.path.join(local_processed_dir, 'X.npy')):
        print(f"[INFO] Found locally processed data in {local_processed_dir}. Using this instead of config path.")
        DATA_PROCESSED_DIR = local_processed_dir

    MODEL_SAVE_PATH = output_config['model_save_path']
    EVALUATION_DIR = output_config['evaluation_dir']
    BATCH_SIZE = evaluation_config['batch_size']
    TTA_ENABLED = evaluation_config['tta_enabled']
    TTA_AUGMENTATIONS = evaluation_config['tta_augmentations']
    IN_CHANNELS = model_config['in_channels']
    USE_HYPERBOLIC = model_config.get('use_hyperbolic', False)
    RANDOM_SEED = config['training']['random_seed']
    VAL_SPLIT = training_config.get('val_split', 0.2)
    NUM_WORKERS = training_config.get('num_workers', 2)
    DROPOUT = model_config.get('dropout', 0.5)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============ SET RANDOM SEEDS ============
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Evaluation on: {DEVICE}")
    print(f"TTA Enabled: {TTA_ENABLED}")
    print(f"{'='*60}\n")
    
    # 1. Load data
    X, y, label_map = load_data(DATA_PROCESSED_DIR)
    num_classes = len(label_map)
    
    # Invert label map: {class_name: idx} -> {idx: class_name}
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(num_classes)]
    
    print(f"Loaded {X.shape[0]} samples with {num_classes} classes")
    
    # 2. Reconstruct test split (identical to training: seed 42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Test set: {len(X_test)} samples\n")
    
    # 3. Create test dataset and dataloader
    test_dataset = TennisDataset(X_test, y_test, augment=False, data_type=MODALITY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 4. Load config to determine model type
    model_config = config.get('model_hyperparameters', {})
    model_type = model_config.get('type', 'HDGCN')
    
    # Check if model checkpoint exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at {MODEL_SAVE_PATH}")
        return
    
    # Auto-detect model type from checkpoint keys
    print(f"Loading checkpoint to detect model architecture...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    
    # Check for distinctive keys to determine model type
    checkpoint_keys = checkpoint.keys()
    has_hdgcn_keys = any('conv_down' in key or 'aha.' in key for key in checkpoint_keys)
    has_ctrgcn_keys = any('alpha' in key or 'convs.' in key for key in checkpoint_keys)
    
    if has_hdgcn_keys and not has_ctrgcn_keys:
        detected_type = 'HDGCN'
    elif has_ctrgcn_keys and not has_hdgcn_keys:
        detected_type = 'CTRGCN'
    else:
        detected_type = model_type  # Use config if can't determine
    
    if detected_type != model_type:
        print(f"WARNING: Config specifies '{model_type}' but checkpoint appears to be '{detected_type}'")
        print(f"Using detected type: {detected_type}")
        model_type = detected_type
    
    print(f"Initializing model type: {model_type}")

    if model_type == 'HDGCN':
        model = HDGCN_Tennis(num_classes=num_classes, in_channels=IN_CHANNELS, drop_out=DROPOUT, use_hyperbolic=USE_HYPERBOLIC)
    elif model_type == 'CTRGCN':
        model = CTRGCN_Tennis(num_classes=num_classes, in_channels=IN_CHANNELS, drop_out=DROPOUT, use_hyperbolic=USE_HYPERBOLIC)
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")
    
    # Load checkpoint with strict=False to handle architecture mismatches
    try:
        model.load_state_dict(checkpoint, strict=True)
        print(f"Model loaded successfully (strict mode)\n")
    except RuntimeError as e:
        print(f"Warning: Could not load model in strict mode. Loading with strict=False...")
        print(f"This may happen if the model architecture has changed.")
        
        # Load with strict=False to ignore missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        print(f"Model loaded successfully (partial load)\n")
    
    model.to(DEVICE)
    model.eval()
    
    # 5. Inference with TTA
    all_preds = []
    all_labels = []
    all_hyp_dists = [] # To store hyperbolic radius
    
    if TTA_ENABLED:
        print(f"Running TTA with {len(TTA_AUGMENTATIONS) + 1} passes (original + {len(TTA_AUGMENTATIONS)} augmentations)...")
    else:
        print("Running standard inference...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs = inputs.to(DEVICE)
            
            if TTA_ENABLED:
                avg_prob = apply_tta(model, inputs, DEVICE, TTA_AUGMENTATIONS)
                # TTA hides the embedding so we do a standard pass just for stats if needed
                if USE_HYPERBOLIC:
                    _, z = model(inputs, return_embeddings=True)
            else:
                if USE_HYPERBOLIC:
                    output, z = model(inputs, return_embeddings=True)
                else:
                    output = model(inputs)
                avg_prob = F.softmax(output, dim=1)
            
            _, predicted = torch.max(avg_prob, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Record hyperbolic distance if available
            if USE_HYPERBOLIC:
                z_norm = torch.norm(z, p=2, dim=1).clamp(max=0.99)
                hyp_dists = torch.arctanh(z_norm).cpu().numpy()
                all_hyp_dists.extend(hyp_dists)
    
    # 6. Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # 7. Print and save results
    print("\n" + "="*60)
    print(f" EVALUATION RESULTS {'(with TTA)' if TTA_ENABLED else '(Standard)'}")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")
    print(report)
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")
    print(report)
    print("="*60)
    
    hyp_report_str = ""
    if USE_HYPERBOLIC and len(all_hyp_dists) > 0:
        print("\n" + "="*60)
        print(" HYPERBOLIC SPACE STATISTICS (Avg Radius per Class)")
        print("="*60)
        hyp_report_str += "\nHYPERBOLIC SPACE STATISTICS (Avg Radius per Class)\n"
        hyp_report_str += "-"*60 + "\n"
        
        # Calculate stats per class
        all_hyp_dists = np.array(all_hyp_dists)
        all_preds_np = np.array(all_preds)
        
        for class_idx in range(num_classes):
            mask = (all_preds_np == class_idx)
            if np.any(mask):
                class_dists = all_hyp_dists[mask]
                avg_dist = np.mean(class_dists)
                max_dist = np.max(class_dists)
                min_dist = np.min(class_dists)
                class_name = class_names[class_idx]
                
                stat_line = f"{class_name:<20} | Avg: {avg_dist:.4f} | Min: {min_dist:.4f} | Max: {max_dist:.4f}"
                print(stat_line)
                hyp_report_str += stat_line + "\n"
        
        overall_avg = np.mean(all_hyp_dists)
        overall_line = f"\nOVERALL AVG RADIUS: {overall_avg:.4f}"
        print(overall_line)
        hyp_report_str += overall_line + "\n"
        print("="*60)
        
    # Save report
    report_path = os.path.join(EVALUATION_DIR, f"report_{'tta' if TTA_ENABLED else 'standard'}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"TTA Enabled: {TTA_ENABLED}\n\n")
        f.write(report)
        if hyp_report_str:
            f.write("\n" + hyp_report_str)
    
    # Save confusion matrix
    cm_title = f"Confusion Matrix - {model_type} {'(with TTA)' if TTA_ENABLED else ''}"
    cm_path = os.path.join(EVALUATION_DIR, f"confusion_matrix_{'tta' if TTA_ENABLED else 'standard'}.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path, title=cm_title)
    
    print(f"\nResults saved to: {EVALUATION_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model with optional Test Time Augmentation (TTA)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()
    
    main(args.config)