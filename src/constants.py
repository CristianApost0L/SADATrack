"""
Constants and mappings for Tennis Swing Classification.
"""

# The 12 official TheTis classes
THETIS_CLASSES = [
    "backhand2hands",
    "backhand", 
    "backhand_slice",
    "backhand_volley",
    "forehand_flat",
    "forehand_openstands",
    "forehand_slice",
    "forehand_volley",
    "flat_service",
    "kick_service",
    "slice_service",
    "smash"
]

# COCO-17 Skeleton Configuration
COCO_NUM_JOINTS = 17
COCO_CENTER_OF_MASS = 11  # Left hip (mid-point between joints 11 and 12)

# COCO Keypoints Indices
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW = 7
KP_R_ELBOW = 8
KP_L_WRIST = 9
KP_R_WRIST = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

# COCO-17 joint names
COCO_JOINT_NAMES = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]

# COCO skeleton connections for visualization (indices)
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs
    (5, 11), (6, 12), (5, 6),                         # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),                  # Arms
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)    # Head
]

# Alias to follow COCO naming convention (same as SKELETON_CONNECTIONS)
COCO_CONNECTIONS = SKELETON_CONNECTIONS

# Mapping from class name to label index
LABEL_MAP = {cls_name: i for i, cls_name in enumerate(THETIS_CLASSES)}

# COCO Keypoints Left/Right pairs (for horizontal flip)
COCO_SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# COCO Keypoint indices for body parts (for local zoom)
COCO_UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # head, shoulders, arms
COCO_LOWER_BODY = [11, 12, 13, 14, 15, 16]  # hips, legs

# Pairs (child, parent) to define bones for COCO-17
# Root is assumed to be 12 (Right Hip) for the tree structure
COCO_BONE_PAIRS = [
    (1, 0), (2, 0), (3, 1), (4, 2),     # Face -> Nose
    (0, 5),                             # Nose -> L.Shoulder
    (5, 11), (7, 5), (9, 7),            # L.Arm/Torso -> L.Hip
    (11, 12),                           # L.Hip -> R.Hip (Root)
    (13, 11), (15, 13),                 # L.Leg -> L.Hip
    (6, 12), (8, 6), (10, 8),           # R.Arm/Torso -> R.Hip
    (14, 12), (16, 14)                  # R.Leg -> R.Hip
]
