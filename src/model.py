"""
Tennis Swing Classification GCN Models.

Provides wrapper classes for HD-GCN and CTR-GCN adapted to COCO-17 skeleton topology.
Both models use hierarchical/spatial graph structures optimized for tennis action recognition.
"""
import sys
import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# Import COCO constants
from .constants import COCO_NUM_JOINTS, COCO_CENTER_OF_MASS

# =============================================================================
# Import Management (Required due to HD-GCN/CTR-GCN module conflicts)
# =============================================================================

def setup_import_env(repo_rel_path):
    """
    Add external GCN repository to Python path.
    
    Args:
        repo_rel_path: Relative path to repository (e.g., 'HD-GCN-main/HD-GCN-main')
    
    Returns:
        Absolute path to repository
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(current_dir, '..', repo_rel_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path

def cleanup_conflicting_modules(modules_to_clear=['graph', 'graph.tools', 'model']):
    """
    Remove cached modules from sys.modules to prevent conflicts.
    
    CRITICAL: HD-GCN and CTR-GCN both have 'graph.tools' with incompatible APIs.
    This function is necessary when switching between frameworks in the same process.
    
    Args:
        modules_to_clear: List of module prefixes to clear from sys.modules
    """
    for mod in list(sys.modules.keys()):
        for target in modules_to_clear:
            if mod == target or mod.startswith(target + '.'):
                if mod in sys.modules:
                    del sys.modules[mod]

# =============================================================================
# COCO-17 Skeleton Topology Definitions
# =============================================================================

def get_groups_coco17_0based():
    """
    Hierarchical grouping of COCO-17 joints (0-indexed).
    
    Groups organized from core (hips) to extremities (hands/feet/head):
    - Group 0: Hips (11, 12)
    - Group 1: Shoulders (5, 6) + Knees (13, 14)
    - Group 2: Nose (0) + Elbows (7, 8) + Ankles (15, 16)
    - Group 3: Eyes (1, 2) + Ears (3, 4) + Wrists (9, 10)
    
    Returns:
        List of 4 joint groups for hierarchical modeling
    """
    groups = []
    groups.append([11, 12])                  # Core: Hips
    groups.append([5, 6, 13, 14])            # Mid: Shoulders + Knees
    groups.append([0, 7, 8, 15, 16])         # Outer: Nose + Elbows + Ankles
    groups.append([1, 2, 3, 4, 9, 10])       # Extremities: Eyes + Ears + Wrists
    return groups

def get_groups_coco17_1based(dataset='Tennis', CoM=None):
    """
    Hierarchical grouping of COCO-17 joints (1-indexed).
    
    Required by HD-GCN which expects 1-based indexing.
    
    Args:
        dataset: Dataset name (unused, kept for HD-GCN compatibility)
        CoM: Center of mass index (unused, kept for HD-GCN compatibility)
    
    Returns:
        List of 4 joint groups with 1-based indices
    """
    groups0 = get_groups_coco17_0based()
    groups1 = [[idx + 1 for idx in group] for group in groups0]
    return groups1

def get_edgeset_coco17_hierarchical():
    """
    Generate hierarchical edge sets for multi-scale graph convolution.
    
    Creates three types of edges at each hierarchy level:
    - Identity: Self-connections within adjacent groups
    - Forward: Connections from inner (core) to outer groups
    - Reverse: Connections from outer to inner groups
    
    Used by HD-GCN for hierarchical feature aggregation.
    
    Returns:
        List of edge sets: [identity, forward, reverse] for each hierarchy level
    """
    groups = get_groups_coco17_0based()
    identity, forward_hierarchy, reverse_hierarchy = [], [], []

    for i in range(len(groups) - 1):
        # Self-links for current and next group
        self_link = groups[i] + groups[i + 1]
        self_link = [(k, k) for k in self_link]
        identity.append(self_link)
        
        # Forward: core -> extremities
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)
        
        # Reverse: extremities -> core
        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])

    return edges

def get_edgeset_coco17_spatial():
    """
    Generate spatial edge sets based on physical skeleton connectivity.
    
    Defines three types of edges:
    - Self-link: Node to itself
    - Inward: From extremities toward center (hips)
    - Outward: From center toward extremities
    
    Used by CTR-GCN for spatial graph convolution.
    
    COCO-17 Joint Order:
    0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 
    5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist,
    11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
    
    Returns:
        Tuple of (self_link, inward, outward) edge lists
    """
    num_node = COCO_NUM_JOINTS
    self_link = [(i, i) for i in range(num_node)]
    
    # Inward edges: Leaf -> Center (toward hips)
    inward = [
        (9, 7), (7, 5), (5, 11),  # L.Wrist -> L.Elbow -> L.Shoulder -> L.Hip
        (10, 8), (8, 6), (6, 12), # R.Wrist -> R.Elbow -> R.Shoulder -> R.Hip
        (15, 13), (13, 11),       # L.Ankle -> L.Knee -> L.Hip
        (16, 14), (14, 12),       # R.Ankle -> R.Knee -> R.Hip
        (11, 12),                 # L.Hip <-> R.Hip (Core connection)
        (0, 5), (0, 6),           # Nose -> L.Shoulder, R.Shoulder
        (1, 0), (2, 0),           # L.Eye -> Nose, R.Eye -> Nose
        (3, 1), (4, 2)            # L.Ear -> L.Eye, R.Ear -> R.Eye
    ]
    
    # Outward edges: Center -> Leaf
    outward = [(j, i) for (i, j) in inward]
    
    return self_link, inward, outward

# =============================================================================
# Graph Definitions
# =============================================================================

class GraphHD:
    """
    Hierarchical graph structure for HD-GCN.
    
    Creates multi-scale adjacency matrices with hierarchical connections
    from core body parts (hips) to extremities (hands, feet, head).
    """
    
    def __init__(self, CoM=None, labeling_mode='spatial'):
        """
        Args:
            CoM: Center of Mass joint index (default: 11 = left hip)
            labeling_mode: 'spatial' for hierarchical edges
        """
        self.num_node = COCO_NUM_JOINTS
        self.CoM = CoM if CoM is not None else COCO_CENTER_OF_MASS
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        """
        Generate adjacency matrix for hierarchical graph convolution.
        
        Args:
            labeling_mode: 'spatial' uses hierarchical edges
            
        Returns:
            Tuple of (adjacency_matrix, center_of_mass_index)
        """
        from graph import tools as graph_tools
        
        if labeling_mode is None:
            return self.A
            
        if labeling_mode == 'spatial':
            edges = get_edgeset_coco17_hierarchical()
            A = graph_tools.get_hierarchical_graph(self.num_node, edges)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' not supported for GraphHD.")
            
        return A, self.CoM


class GraphCTR:
    """
    Spatial graph structure for CTR-GCN.
    
    Creates adjacency matrices based on physical skeleton connectivity
    with self, inward (toward center), and outward (toward extremities) edges.
    """
    
    def __init__(self, labeling_mode='spatial'):
        """
        Args:
            labeling_mode: 'spatial' for physical skeleton edges
        """
        self.num_node = COCO_NUM_JOINTS
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        """
        Generate adjacency matrix for spatial graph convolution.
        
        Args:
            labeling_mode: 'spatial' uses physical skeleton edges
            
        Returns:
            Adjacency matrix tensor
        """
        from graph import tools as graph_tools
        
        if labeling_mode is None:
            return self.A
            
        if labeling_mode == 'spatial':
            self_link, inward, outward = get_edgeset_coco17_spatial()
            A = graph_tools.get_spatial_graph(self.num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode '{labeling_mode}' not supported for GraphCTR.")
            
        return A


# =============================================================================
# Base Model Class
# =============================================================================

class BaseTennisGCN(nn.Module, ABC):
    """
    Abstract base class for Tennis GCN models.
    
    Provides common functionality:
    - Input dimension handling (ensure 5D: N, C, T, V, M)
    - Channel slicing (match model's expected in_channels)
    - Forward pass template
    """
    
    def __init__(self, num_classes, in_channels=3, drop_out=0, data_type='joint'):
        """
        Args:
            num_classes: Number of tennis swing classes
            in_channels: Input channels (3 for x,y,conf or 4 for x,y,z,conf)
            drop_out: Dropout rate
            data_type: 'joint' or 'bone' (for logging/documentation only)
        """
        super(BaseTennisGCN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.drop_out = drop_out
        self.data_type = data_type
        self.model = None  # Set by subclass
    
    @abstractmethod
    def _build_model(self):
        """Subclass must implement this to build self.model"""
        pass
    
    def __repr__(self):
        """String representation for logging."""
        return (f"{self.__class__.__name__}("
                f"num_classes={self.num_classes}, "
                f"in_channels={self.in_channels}, "
                f"data_type='{self.data_type}')")
    
    def get_model_name(self):
        """
        Get descriptive model name for checkpoint saving.
        
        Returns:
            String like 'hdgcn_joint' or 'ctrgcn_bone'
        """
        model_type = self.__class__.__name__.replace('_Tennis', '').lower()
        return f"{model_type}_{self.data_type}"
    
    def forward(self, x):
        """
        Forward pass with automatic input formatting.
        
        Args:
            x: Input tensor (N, C, T, V) or (N, C, T, V, M)
            
        Returns:
            Class logits (N, num_classes)
        """
        # Ensure 5D input: (N, C, T, V, M)
        if x.dim() == 4:
            x = x.unsqueeze(-1)  # Add person dimension
            
        # Slice channels if input has more than expected
        if x.shape[1] > self.in_channels:
            x = x[:, :self.in_channels, :, :, :]
            
        return self.model(x)
# =============================================================================
# HD-GCN Model
# =============================================================================

class HDGCN_Tennis(BaseTennisGCN):
    """
    HD-GCN (Hierarchical Directed Graph Convolutional Network) for Tennis.
    
    Uses hierarchical graph structure to model relationships between body parts
    at multiple scales (core -> extremities). Effective for capturing complex
    motion patterns in tennis swings.
    
    Architecture:
    - Multi-scale hierarchical convolutions
    - Directed edges (forward/reverse hierarchy)
    - Center of Mass (CoM) as structural anchor
    
    Supports both joint and bone representations:
    - Joint: Absolute positions of skeleton keypoints
    - Bone: Relative vectors between parent-child joints
    
    Reference: 
    "Hierarchical Graph Convolutional Networks for Action Recognition"
    HD-GCN Paper
    """
    
    def __init__(self, num_classes, in_channels=3, drop_out=0, data_type='joint', **kwargs):
        """
        Args:
            num_classes: Number of tennis swing classes (e.g., 12 for TheTis)
            in_channels: Input channels (3: x,y,conf or 4: x,y,z,conf)
            drop_out: Dropout rate for regularization
            data_type: 'joint' or 'bone' representation (for logging/documentation)
            **kwargs: Additional arguments (ignored, kept for compatibility)
        """
        super(HDGCN_Tennis, self).__init__(num_classes, in_channels, drop_out, data_type)
        self._build_model()
    
    def _build_model(self):
        """Initialize HD-GCN model with COCO-17 hierarchy."""
        # Setup HD-GCN environment
        setup_import_env(os.path.join('HD-GCN-main', 'HD-GCN-main'))
        
        # Import with conflict handling
        try:
            import model.HDGCN as OfficialHDGCN
        except ImportError:
            # Retry after cleaning conflicting modules
            cleanup_conflicting_modules()
            import model.HDGCN as OfficialHDGCN
        
        # Monkey patch: Replace get_groups with COCO-17 version
        # HD-GCN uses this function to determine hierarchical groupings
        OfficialHDGCN.get_groups = get_groups_coco17_1based
        
        # Graph configuration
        graph_args = {
            'CoM': COCO_CENTER_OF_MASS,  # Mid-point between hips
            'labeling_mode': 'spatial'    # Use hierarchical edges
        }
        
        # Build model
        self.model = OfficialHDGCN.Model(
            num_class=self.num_classes,
            num_point=COCO_NUM_JOINTS,
            num_person=1,                     # Single player tracking
            graph='src.model.GraphHD',        # Reference to our GraphHD
            graph_args=graph_args,
            in_channels=self.in_channels,
            drop_out=self.drop_out
        )


# =============================================================================
# CTR-GCN Model
# =============================================================================

class CTRGCN_Tennis(BaseTennisGCN):
    """
    CTR-GCN (Channel-wise Topology Refinement GCN) for Tennis.
    
    Uses spatial graph structure based on physical skeleton connectivity.
    Learns channel-specific topology refinements for better feature extraction.
    
    Architecture:
    - Spatial graph convolution (physical skeleton edges)
    - Channel-wise topology refinement
    - Adaptive graph learning
    
    Supports both joint and bone representations:
    - Joint: Absolute positions of skeleton keypoints
    - Bone: Relative vectors between parent-child joints
    
    Reference:
    "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition"
    CTR-GCN Paper
    """
    
    def __init__(self, num_classes, in_channels=3, drop_out=0, data_type='joint', **kwargs):
        """
        Args:
            num_classes: Number of tennis swing classes (e.g., 12 for TheTis)
            in_channels: Input channels (3: x,y,conf or 4: x,y,z,conf)
            drop_out: Dropout rate for regularization
            data_type: 'joint' or 'bone' representation (for logging/documentation)
            **kwargs: Additional arguments (ignored, kept for compatibility)
        """
        super(CTRGCN_Tennis, self).__init__(num_classes, in_channels, drop_out, data_type)
        self._build_model()
    
    def _build_model(self):
        """Initialize CTR-GCN model with COCO-17 spatial graph."""
        # Setup CTR-GCN environment
        setup_import_env(os.path.join('CTR-GCN-main', 'CTR-GCN-main'))
        
        # CRITICAL: Clean modules to prevent conflicts with HD-GCN
        # Both frameworks have 'graph.tools' with incompatible functions
        cleanup_conflicting_modules()
        
        import model.ctrgcn as OfficialCTRGCN
        
        # Graph configuration
        graph_args = {
            'labeling_mode': 'spatial'  # Use physical skeleton edges
        }
        
        # Build model
        self.model = OfficialCTRGCN.Model(
            num_class=self.num_classes,
            num_point=COCO_NUM_JOINTS,
            num_person=1,                     # Single player tracking
            graph='src.model.GraphCTR',       # Reference to our GraphCTR
            graph_args=graph_args,
            in_channels=self.in_channels,
            drop_out=self.drop_out,
            adaptive=True                     # Enable adaptive graph learning
        )
