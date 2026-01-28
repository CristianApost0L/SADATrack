import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import TennisDataset
from src.model import HDGCN_Tennis

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

def train(config_path):
    # Load configuration
    config = load_config(config_path)
    data_config = config['data']
    training_config = config['training']
    model_config = config['model_hyperparameters']
    output_config = config['output']
    
    BATCH_SIZE = training_config['batch_size']
    EPOCHS = training_config['epochs']
    LEARNING_RATE = training_config['learning_rate']
    WEIGHT_DECAY = training_config['weight_decay']
    VAL_SPLIT = training_config['val_split']
    RANDOM_SEED = training_config['random_seed']
    NUM_WORKERS = training_config['num_workers']
    EARLY_STOPPING_PATIENCE = training_config.get('early_stopping_patience', 10)
    CHECKPOINT_FREQUENCY = training_config.get('checkpoint_frequency', 5)
    MONITOR_METRIC = training_config.get('monitor_metric', 'f1')
    
    DROPOUT = model_config['dropout']
    IN_CHANNELS = model_config['in_channels']
    DATA_PROCESSED_DIR = data_config['processed_dir']
    MODEL_SAVE_PATH = output_config['model_save_path']
    PLOTS_OUTPUT_DIR = output_config['plots_output_dir']
    CHECKPOINTS_DIR = output_config.get('checkpoints_dir', 'models/checkpoints')
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============ SET RANDOM SEEDS ============
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Training started on: {DEVICE}")
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
    
    # 2. Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    
    # Calculate class weights to handle potential imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    print(f"\nClass weights (for imbalanced datasets):")
    for idx, weight in enumerate(class_weights):
        print(f"  Class {idx}: {weight:.4f}")
    
    # 3. Create datasets and dataloaders
    # Augmentation ONLY on training set
    train_dataset = TennisDataset(X_train, y_train, augment=True)
    val_dataset = TennisDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 4. Initialize model
    model = HDGCN_Tennis(num_classes=num_classes, in_channels=IN_CHANNELS, drop_out=DROPOUT)
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': []}
    best_f1 = 0.0
    patience_counter = 0
    
    # Create directories based on config
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
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
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
        
        # Update learning rate
        scheduler.step()
        
        # Log and save
        print(f"End Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
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
        
        # Save best model (based on F1 score)
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> New best model saved! (F1: {best_f1:.4f}, Acc: {val_acc:.4f})")
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
            
    # Training complete
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Validation F1-Score: {best_f1:.4f}")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
    print(f"{'='*60}")
    save_plots(history, PLOTS_OUTPUT_DIR)
    print(f"Plots saved in '{PLOTS_OUTPUT_DIR}/' folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HDGCN model for swing classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file (default: config.yaml)')
    args = parser.parse_args()
    
    train(args.config)