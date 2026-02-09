import os
import sys
import yaml
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import TennisDataset, CurriculumLearningScheduler
from src.model import HDGCN_Tennis, CTRGCN_Tennis

def load_config(config_path):
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_plots(history, output_dir='runs'):
    """Save training plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()

    # Plot F1 Score
    if 'val_f1' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_f1'], label='Train F1')
        plt.plot(history['val_f1'], label='Val F1')
        plt.title('Model F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'f1_score.png'))
        plt.close()
    
    # Plot Augmentation Strength (Curriculum Learning)
    if 'augmentation_strength' in history and len(history['augmentation_strength']) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history['augmentation_strength'], label='Augmentation Strength', color='orange')
        plt.axhline(y=0.25, color='r', linestyle='--', label='Weak', alpha=0.7)
        plt.axhline(y=0.5, color='y', linestyle='--', label='Medium', alpha=0.7)
        plt.axhline(y=1.0, color='g', linestyle='--', label='Strong', alpha=0.7)
        plt.title('Curriculum Learning: Augmentation Strength Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Augmentation Strength')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(os.path.join(output_dir, 'augmentation_strength.png'))
        plt.close()

def run_training_fold(X_train, y_train, X_val, y_val, config, fold_idx=None, num_classes=None, device=None):
    """Run training for a single fold or single train/val split"""

    # Extract configurations
    training_config = config['training']
    model_config = config['model_hyperparameters']
    output_config = config['output']
    
    # Training hyperparameters
    BATCH_SIZE = training_config['batch_size']
    EPOCHS = training_config['epochs']
    LEARNING_RATE = training_config['learning_rate']
    WEIGHT_DECAY = training_config['weight_decay']
    DROPOUT = model_config['dropout']
    IN_CHANNELS = model_config['in_channels']
    RANDOM_SEED = training_config['random_seed']
    NUM_WORKERS = training_config['num_workers']
    EARLY_STOPPING_PATIENCE = training_config.get('early_stopping_patience', 10)
    CHECKPOINT_FREQUENCY = training_config.get('checkpoint_frequency', 5)

    # Type of skeleton data (joint, bone)
    MODALITY = training_config.get('modality', 'joint')

    # Model type (HDGCN, CTRGCN, etc.)
    MODEL_TYPE = model_config.get('type', 'HDGCN')
    
    # Curriculum learning settings
    USE_CURRICULUM = training_config.get('use_curriculum_learning', False)
    CURRICULUM_SCHEDULE = training_config.get('curriculum_schedule', 'linear')
    
    # Augmentation settings
    USE_AUGMENTATION = training_config.get('augment', True)
    DEFAULT_AUG_STRENGTH = training_config.get('default_augmentation_strength', 'medium')
    
    # Modify paths if k-fold
    if fold_idx is not None:
        fold_base = output_config.get('folds_output_dir', 'runs/kfold_results')
        model_name, ext = os.path.splitext(output_config['model_save_path'])
        
        current_model_type = MODEL_TYPE
        filename = f'best_model_{current_model_type.lower()}{ext}'
        
        MODEL_SAVE_PATH = os.path.join(fold_base, f'fold_{fold_idx}', filename)
        PLOTS_OUTPUT_DIR = os.path.join(fold_base, f'fold_{fold_idx}', 'plots')
        CHECKPOINTS_DIR = os.path.join(fold_base, f'fold_{fold_idx}', 'checkpoints')
        print(f"\n{'='*20} Starting Fold {fold_idx} {'='*20}")
    else:
        MODEL_SAVE_PATH = output_config['model_save_path']
        PLOTS_OUTPUT_DIR = output_config['plots_output_dir']
        CHECKPOINTS_DIR = output_config.get('checkpoints_dir', 'models/checkpoints')
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"\nClass weights (for imbalanced datasets):")
    for idx, weight in enumerate(class_weights):
        print(f"  Class {idx}: {weight:.4f}")
    
    # Create datasets and dataloaders
    train_dataset = TennisDataset(X_train, y_train, augment=USE_AUGMENTATION, data_type=MODALITY)
    val_dataset = TennisDataset(X_val, y_val, augment=False, data_type=MODALITY)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Create handedness robustness loader (forced flip)
    val_handedness_loader = None
    if USE_AUGMENTATION:
        val_dataset_handedness = TennisDataset(X_val, y_val, augment=False, data_type=MODALITY, force_flip_val=True)
        val_handedness_loader = DataLoader(val_dataset_handedness, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"\n[INFO] Dataset Augmentation Configuration:")
    if USE_AUGMENTATION:
        print(f"       Status: ENABLED")
        print(f"       - Geometric:")
        print(f"         * Horizontal Flip:    {train_dataset.aug_probs.get('flip_prob', 0.0):.2f} (prob)")
        print(f"         * Global Rotation:    +/- {train_dataset.aug_probs.get('rotation_range', 0)} degrees")
        print(f"         * Global Scaling:     +/- {train_dataset.aug_probs.get('scale_range', 0.0):.2f}")
        print(f"         * Shearing:           {train_dataset.aug_probs.get('apply_shearing', False)}")
        print(f"         * Local Zoom:         {train_dataset.aug_probs.get('apply_local_zoom', False)}")
        
        print(f"       - Temporal:")
        print(f"         * Temporal Crop:      {train_dataset.aug_probs.get('apply_temporal_crop', False)}")
        print(f"         * Temporal Scaling:   {train_dataset.aug_probs.get('apply_temporal_scaling', False)}")
        print(f"         * Frame Dropping:     {train_dataset.aug_probs.get('apply_frame_dropping', False)}")
        
        print(f"       - Noise/Robustness:")
        print(f"         * Gaussian Noise:     Std {train_dataset.aug_probs.get('noise_std', 0.0)}")
        print(f"         * Local Jitter:       {train_dataset.aug_probs.get('apply_local_jitter', False)}")
        print(f"         * Bone Scaling:       {train_dataset.aug_probs.get('apply_bone_scaling', False)}")
        print(f"         * Keypoint Dropout:   {train_dataset.aug_probs.get('apply_keypoint_dropout', False)}")
        print(f"         * Confidence Mask:    {train_dataset.aug_probs.get('apply_confidence_mask', False)}")
        print(f"         * Confidence Jitter:  {train_dataset.aug_probs.get('apply_confidence_jitter', False)}")
    else:
        print(f"       Status: DISABLED")
    
    # Initialize curriculum learning scheduler if enabled
    curriculum_scheduler = None
    if USE_CURRICULUM:
        curriculum_scheduler = CurriculumLearningScheduler(EPOCHS, schedule_type=CURRICULUM_SCHEDULE)
        print(f"\n[INFO] Curriculum Learning ENABLED")
        print(f"       Schedule type: {CURRICULUM_SCHEDULE}")
        print(f"       Augmentation will progressively increase from weak -> strong")
    else:
        train_dataset.set_augmentation_strength(DEFAULT_AUG_STRENGTH)
        print(f"\n[INFO] Curriculum Learning DISABLED")
        print(f"       Using {DEFAULT_AUG_STRENGTH} augmentation strength")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    
    # Initialize model
    model_type = MODEL_TYPE
    print(f"\n[INFO] Initializing model: {model_type}")
    
    if model_type == 'HDGCN':
        model = HDGCN_Tennis(num_classes=num_classes, in_channels=IN_CHANNELS, drop_out=DROPOUT)
    elif model_type == 'CTRGCN':
        model = CTRGCN_Tennis(num_classes=num_classes, in_channels=IN_CHANNELS, drop_out=DROPOUT)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': [], 'augmentation_strength': []}
    best_f1 = 0.0
    patience_counter = 0
    
    # Create directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        # --- UPDATE AUGMENTATION STRENGTH (Curriculum Learning) ---
        if curriculum_scheduler is not None:
            # Update scheduler state
            curriculum_scheduler.step(epoch)
            
            # Log milestone when strength changes
            if curriculum_scheduler.just_changed():
                strength = curriculum_scheduler.get_current_strength()
                aug_params = curriculum_scheduler.get_augmentation_params(epoch)
                print(f"\n{'='*60}")
                print(f"[CURRICULUM] Epoch {epoch+1}: Switching to {strength.upper()} augmentation")
                print(f"             flip_prob={aug_params['flip_prob']:.2f}, "
                      f"rotation={aug_params['rotation_range']}°, "
                      f"scale=±{aug_params['scale_range']:.2f}, "
                      f"noise_std={aug_params['noise_std']:.4f}")
                print(f"{'='*60}\n")
            
            # Apply augmentation params
            aug_params = curriculum_scheduler.get_augmentation_params(epoch)
            train_dataset.update_augmentation_params(aug_params)
            current_strength = (aug_params['flip_prob'] + aug_params['rotation_range']/20 + aug_params['scale_range']/0.2 + aug_params['noise_std']/0.01) / 4
            history['augmentation_strength'].append(current_strength)
            strength_info = f" | Aug Strength: {current_strength:.3f}"
        else:
            strength_info = ""
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        fold_info = f" [Fold {fold_idx}]" if fold_idx is not None else ""
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]{fold_info}{strength_info}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Calculate training metrics
        all_preds_train = []
        all_labels_train = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds_train.extend(predicted.cpu().numpy())
                all_labels_train.extend(labels.cpu().numpy())
        model.train()
        
        train_precision = precision_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
        train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted', zero_division=0)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds_val = []
        all_labels_val = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds_val.extend(predicted.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Calculate validation metrics
        val_precision = precision_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted', zero_division=0)
        
        val_handedness_f1 = None
        
        # --- HANDEDNESS ROBUSTNESS (Left-Handed Simulation) ---
        # Test performance on FLIPPED validation data to ensure the model generalizes to left-handed players
        if val_handedness_loader is not None:
            all_preds_handedness = []
            all_labels_handedness = []
            with torch.no_grad():
                for inputs, labels in val_handedness_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    all_preds_handedness.extend(predicted.cpu().numpy())
                    all_labels_handedness.extend(labels.cpu().numpy())
            
            val_handedness_f1 = f1_score(all_labels_handedness, all_preds_handedness, average='weighted', zero_division=0)
        
        # Combined robust score (handedness only, back view removed)
        final_robust_score = val_f1
        if val_handedness_f1 is not None:
            final_robust_score = (val_f1 + val_handedness_f1) / 2.0

        # Update learning rate
        scheduler.step()
        
        # Log and save
        handedness_str = f"{val_handedness_f1:.4f}" if val_handedness_f1 is not None else "N/A"
        print(f"End Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Handedness(Flip): {handedness_str} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        # Combined Score for Best Model Selection
        # We weigh all robustness metrics (back view + handedness)
        combined_score = final_robust_score
        
        # Save best model
        if combined_score > best_f1:
            best_f1 = combined_score
            patience_counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            handedness_log = f", Flip:{val_handedness_f1:.4f}" if val_handedness_f1 is not None else ""
            print(f"--> New best model saved! (Comb: {combined_score:.4f} [Val:{val_f1:.4f}{handedness_log}])")
        else:
            patience_counter += 1
        
        # Save checkpoint periodically
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'history': history
            }, checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n --- Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break
            
    # Training complete for this fold
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx if fold_idx is not None else 'Single'} completed!")
    print(f"Best Validation F1-Score: {best_f1:.4f}")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"{'='*60}")
    save_plots(history, PLOTS_OUTPUT_DIR)
    print(f"Plots saved in '{PLOTS_OUTPUT_DIR}/' folder")
    
    return best_f1, history

def train(config_path, k_folds=None):
    # Load configuration
    config = load_config(config_path)
    data_config = config['data']
    training_config = config['training']
    
    RANDOM_SEED = training_config['random_seed']
    VAL_SPLIT = training_config['val_split']
    DATA_PROCESSED_DIR = data_config['processed_dir']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============ SET RANDOM SEEDS ============
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Training started on: {DEVICE}")
    
    local_processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    if os.path.exists(os.path.join(local_processed_dir, 'X.npy')):
        print(f"[INFO] Found locally processed data in {local_processed_dir}. Using this instead of config path.")
        DATA_PROCESSED_DIR = local_processed_dir
        
    X_path = os.path.join(DATA_PROCESSED_DIR, 'X.npy')
    y_path = os.path.join(DATA_PROCESSED_DIR, 'y.npy')
    label_map_path = os.path.join(DATA_PROCESSED_DIR, 'label_map.npy')
    
    if not os.path.exists(X_path):
        print(f"ERROR: Data not found. Run 'scripts/prepare_data.py' first")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()
    num_classes = len(label_map)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {num_classes} classes")
    
    # Check if using k-fold cross validation
    if k_folds and k_folds > 1:
        print(f"\n{'='*60}")
        print(f"Running {k_folds}-Fold Cross Validation")
        print(f"{'='*60}")
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            best_f1, _ = run_training_fold(X_train, y_train, X_val, y_val, config, 
                                          fold_idx=fold+1, num_classes=num_classes, device=DEVICE)
            fold_f1_scores.append(best_f1)
        
        # Print k-fold results
        print(f"\n{'='*60}")
        print(f"K-Fold Cross Validation Results ({k_folds} Folds):")
        print(f"{'='*60}")
        for fold_idx, f1 in enumerate(fold_f1_scores):
            print(f"  Fold {fold_idx+1}: F1 = {f1:.4f}")
        print(f"  Average F1: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
        print(f"{'='*60}")
        
    else:
        # Single train/val split
        print(f"\nRunning Single Train/Val Split")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y
        )
        run_training_fold(X_train, y_train, X_val, y_val, config, 
                         fold_idx=None, num_classes=num_classes, device=DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Tennis Swing Classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    parser.add_argument('--kfold', type=int, default=None,
                        help='Number of folds for K-Fold Cross Validation (optional, e.g., 5)')
    args = parser.parse_args()
    
    train(args.config, k_folds=args.kfold)