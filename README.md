# Tennis Swing Classifier

**Advanced Graph Convolutional Network-based Action Recognition System for Tennis Stroke Classification**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Configuration](#configuration)
- [Future Work](#future-work)

---

## 🎯 Overview

This project implements a state-of-the-art tennis swing classification system using Graph Convolutional Networks (GCNs) on 2D skeletal data extracted from RGB videos. The system automatically detects and classifies various tennis strokes from the THETIS dataset with high accuracy using spatial-temporal graph modeling.

**Key Highlights:**
- 🎾 **17 Tennis Action Classes** (forehand, backhand, serve, volley, etc.)
- 🤖 **Dual GCN Architectures**: HD-GCN and CTR-GCN
- 📹 **End-to-End Pipeline**: From video to classification
- 🎯 **Smart Cropping**: Automatic player detection and tracking
- 🔄 **Robust Normalization**: Torso-based pose normalization for scale/position invariance
- 📊 **Test-Time Augmentation**: Enhanced inference robustness

---

## ✨ Features

### Core Capabilities
- **Automatic Pose Extraction**: YOLO-Pose 26 for real-time 2D keypoint detection
- **Smart Crop & Tracking**: Dynamic frame cropping focused on main player
- **Graph-Based Modeling**: Hierarchical (HD-GCN) and spatial (CTR-GCN) graph convolutions
- **Data Augmentation Pipeline**: 13 augmentation types including temporal, spatial, and confidence-based
- **Curriculum Learning**: Progressive augmentation strength scheduling
- **K-Fold Cross-Validation**: Robust performance evaluation
- **Test-Time Augmentation (TTA)**: Ensemble predictions from multiple augmented views

### Technical Innovations
- **Unified Normalization**: Robust skeleton normalization across training and inference
- **COCO-17 Adaptation**: Custom graph topology for COCO skeleton format
- **Smart Cropping**: Re-detection on cropped regions for improved keypoint accuracy
- **Flexible Architecture**: Easy model switching (HD-GCN ↔ CTR-GCN)

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

Video (RGB) → YOLO-Pose → Smart Crop → Keypoints (17x3)
                                              ↓
                                    Normalize (Torso-based)
                                              ↓
                                    Sequence (40 frames)
                                              ↓
                              ┌───────────────┴───────────────┐
                              │                               │
                          HD-GCN                          CTR-GCN
                    (Hierarchical Graph)            (Spatial Graph)
                              │                               │
                              └───────────────┬───────────────┘
                                              ↓
                                    Softmax Classification
                                              ↓
                                    12 Tennis Actions

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

Video → Pose Extraction → Normalize → GCN Model → Predictions
                              ↓
                    [Optional TTA]
                    Scale + Noise Variants
                              ↓
                    Ensemble Averaging
```

### Graph Construction

**COCO-17 Skeleton Topology**:
- **Nodes**: 17 body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Edges**: Physical bone connections + hierarchical groupings
- **Center of Mass**: Left hip (joint 11) for hierarchical modeling

**Hierarchical Groups** (HD-GCN):
```
Level 0: [11, 12]                    # Hips (core)
Level 1: [5, 6, 13, 14]              # Shoulders + Knees
Level 2: [0, 7, 8, 15, 16]           # Nose + Elbows + Ankles
Level 3: [1, 2, 3, 4, 9, 10]         # Eyes + Ears + Wrists
```

---

## 📊 Dataset

**THETIS Tennis Dataset**
- **Source**: RGB videos of tennis strokes
- **Classes**: 12 action categories

**Our Processing**:
- Extract 2D poses using YOLO-Pose 26
- Normalize sequences to 40 frames
- Apply robust torso-based normalization

---

## 🚀 Installation

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/your-username/Swing_Classifier.git
cd Swing_Classifier
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLO26-Pose Model**

5. **Prepare Dataset**
- Place THETIS dataset in `archive(1)/VIDEO_RGB/`
- Update paths in `config.yaml` if needed

---

## 💻 Usage

### 1. Data Preparation

Extract poses from videos and apply normalization:

```bash
python scripts/prepare_data.py
```

**Output**: `data/processed/X.npy`, `y.npy`, `label_map.npy`

---

### 2. Training

**Single Train/Val Split**:
```bash
python scripts/train.py --config config.yaml
```

**K-Fold Cross-Validation**:
```bash
python scripts/train.py --config config.yaml --kfold 5
```

**Key Training Features**:
- Automatic class weight balancing
- Curriculum learning (optional)

**Saved Outputs**:
- Best model: `models/best_model_{hdgcn|ctrgcn}.pth`
- Training plots: `runs/training_*.png`
- Checkpoints: `models/checkpoints/epoch_*.pth`

---

### 3. Evaluation

**Standard Evaluation**:
```bash
python scripts/evaluate_tta.py --config config.yaml
```

**Outputs**:
- Confusion matrix: `runs/evaluation_tta/confusion_matrix.png`
- Classification report: `runs/evaluation_tta/classification_report.txt`
- Per-class metrics

---

### 4. Demo (Real-time Inference)

Classify actions from new videos:

```bash
python scripts/demo.py --video path/to/video.mp4 --model models/best_model_ctrgcn.pth
```

**Features**:
- Frame-by-frame pose visualization
- Real-time classification
- Output video with predictions overlaid
- Confidence scores display

---

### 5. Validation Tools

**Validate Pose Extraction**:
```bash
python scripts/validate_extraction.py --mode video --video path/to/video.mp4
```

**Visualize**:
- Raw keypoints on video frames
- Normalized skeleton graphs
- Hip centering verification
- Torso length consistency

---

## 📁 Project Structure

```
Swing_Classifier/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
│
├── scripts/                    # Executable scripts
│   ├── prepare_data.py         # Pose extraction & normalization
│   ├── train.py                # Model training
│   ├── evaluate_tta.py         # Evaluation with TTA
│   ├── demo.py                 # Real-time inference
│   ├── validate_extraction.py  # Pose extraction validation
│   └── analyze_dataset.py      # Dataset statistics
│
├── src/                        # Core library
│   ├── constants.py            # COCO skeleton definitions
│   ├── extractor.py            # PoseExtractor (YOLO-Pose wrapper)
│   ├── normalization.py        # Robust skeleton normalization
│   ├── dataset.py              # TennisDataset + augmentations
│   ├── model.py                # GCN model wrappers
│   ├── curriculum.py           # Curriculum learning scheduler
│   └── utils.py                # Helper functions
│
├── models/                     # Saved models & checkpoints
│   ├── yolov26x-pose.pt        # YOLO-Pose weights
│   └── best_model_*.pth        # Trained GCN models
│
├── data/                       # Processed data
│   └── processed/
│       ├── X.npy               # Pose sequences (N, T, V, C)
│       ├── y.npy               # Labels
│       └── label_map.npy       # Class mapping
│
├── runs/                       # Training & evaluation outputs
│   ├── training_*.png          # Loss/accuracy plots
│   ├── evaluation_tta/         # Evaluation results
│   └── validation/             # Validation videos
│
├── CTR-GCN-main/              # CTR-GCN reference implementation
└── HD-GCN-main/               # HD-GCN reference implementation
```

---

## 🧠 Models

### HD-GCN (Hierarchical Graph Convolutional Network)

**Architecture**:
- Multi-scale hierarchical graph structure
- 4 hierarchical levels (core → extremities)
- Adaptive graph learning
- Attention-based feature aggregation (AHA modules)

**Key Features**:
- Captures body part hierarchies
- Effective for complex full-body motions
- Higher parameter count

**Use Cases**: Complex strokes with full-body coordination (serve, overhead)

---

### CTR-GCN (Channel-wise Topology Refinement GCN)

**Architecture**:
- Dynamic channel-wise graph topology
- Physical bone-based connectivity
- Topology refinement per channel

**Key Features**:
- Adaptive spatial modeling
- Lower computational cost
- Strong baseline performance

**Use Cases**: General-purpose, faster inference

---

## ⚙️ Configuration

Key parameters in `config.yaml`:

```yaml
# Model Selection
model_hyperparameters:
  type: 'CTRGCN'  # or 'HDGCN'
  in_channels: 3  # (x, y, confidence)
  dropout: 0.5

# Training
training:
  batch_size: 32
  epochs: 60
  learning_rate: 0.001
  val_split: 0.2
  random_seed: 42
  
  # Augmentation
  augment: true
  use_curriculum_learning: false
  
  # Modality
  modality: 'joint'  # or 'bone'

# Evaluation
evaluation:
  tta_enabled: true
  tta_augmentations:
    - scale: 0.95
      noise: 0.01
    - scale: 1.05
      noise: 0.01
```

## 🔮 Future Work

### 1. Multi-CoM Ensemble Architecture

**Motivation**: The current implementation uses a single Center of Mass (left hip, joint 11) for HD-GCN graph construction, which introduces geometric asymmetry. Leveraging multiple CoMs can create complementary representations.

#### 1.1 Symmetric Multi-CoM Ensemble

**Approach**:
- Train 3 separate HD-GCN models with different CoMs:
  - **Model A**: Left Hip (11) - current baseline
  - **Model B**: Right Hip (12) - mirror bias
  - **Model C**: Virtual Hip Center - perfect symmetry

**Ensemble Strategy**:
```python
# Late fusion via weighted averaging
prediction = (w_A * pred_A + w_B * pred_B + w_C * pred_C)
```

**Expected Benefits**:
- Eliminate left/right bias
- Capture both asymmetric and symmetric features
- Robust to player handedness

---

#### 1.2 Handedness-Aware Adaptive Ensemble

**Motivation**: Different tennis strokes have dominant hand preferences:
- Forehand → dominant hand leads
- Backhand → non-dominant hand leads
- Serve → dominant hand critical

**Approach**:

**Phase 1: Handedness Detection**
```python
def detect_handedness(skeleton_sequence):
    """
    Infer dominant hand from wrist motion energy
    """
    left_wrist_motion = compute_motion_energy(seq[:, 9, :2])   # Joint 9
    right_wrist_motion = compute_motion_energy(seq[:, 10, :2]) # Joint 10
    
    return 'right' if right_wrist_motion > left_wrist_motion else 'left'
```

**Phase 2: Bias-CoM Models**
- **Left-Biased CoM**: Center on left shoulder (5) + left hip (11) midpoint
- **Right-Biased CoM**: Center on right shoulder (6) + right hip (12) midpoint

**Phase 3: Dynamic Ensemble**
```python
def adaptive_ensemble(skeleton_seq):
    handedness = detect_handedness(skeleton_seq)
    
    if handedness == 'right':
        # Weight right-biased model higher
        weights = {'left_CoM': 0.2, 'right_CoM': 0.6, 'center_CoM': 0.2}
    else:
        weights = {'left_CoM': 0.6, 'right_CoM': 0.2, 'center_CoM': 0.2}
    
    return weighted_avg(predictions, weights)
```

**Expected Benefits**:
- Optimal graph topology per stroke type
- Exploit inherent asymmetry when beneficial
- Better generalization across player styles

---

#### 1.3 Multi-Scale Hierarchical Ensemble

**Approach**: Combine models with CoMs at different hierarchical levels

**Model Variants**:
1. **Core-CoM**: Hip center (current)
2. **Torso-CoM**: Shoulder midpoint (joint 5-6 center)
3. **Full-Body-CoM**: Geometric center of all joints

**Hierarchical Fusion**:
```python
# Early fusion: Concatenate features before classification
features = torch.cat([
    model_core.get_features(x),    # Fine-grained lower body
    model_torso.get_features(x),   # Upper body emphasis
    model_full.get_features(x)     # Global context
], dim=1)

prediction = classifier(features)
```

**Expected Benefits**:
- Multi-scale spatial reasoning
- Capture both local (hand) and global (full-body) patterns
- Robust to occlusions

---

### 2. Additional Future Directions

#### 2.1 Temporal Modeling Enhancements
- **Transformer-based attention** for long-range temporal dependencies
- **Adaptive frame sampling** based on motion saliency
- **Multi-resolution temporal graphs** (slow/fast pathways)

#### 2.2 Multi-Modal Fusion
- **RGB + Skeleton fusion** via late or mid-level fusion
- **Audio integration** for impact sound detection (serve, volley)
- **Court line detection** for spatial context

#### 2.3 Real-Time Deployment
- **Model quantization** (INT8) for edge devices
- **Knowledge distillation** to compact models
- **ONNX/TensorRT** optimization for production

#### 2.4 Few-Shot Learning
- **Meta-learning** for rapid adaptation to new stroke variants
- **Synthetic data generation** via pose simulator
- **Transfer learning** from larger action recognition datasets

**References**:
- CTR-GCN: Chen et al., "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition", ICCV 2021
- HD-GCN: Lee et al., "Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition", CVPR 2022
- YOLO-Pose: Ultralytics YOLOv26-Pose, 2026

## 🙏 Acknowledgments

- **THETIS Dataset** creators for providing tennis action videos
- **Ultralytics** for YOLO-Pose implementation
- **Original HD-GCN and CTR-GCN** authors for foundational architectures
