import sys
import os
import torch
import torch.nn as nn
import numpy as np
import constants

# Helper to manage imports
def setup_import_env(repo_path):
    """
    Adds an absolute path to sys.path so we can import modules from it.
    """
    if not os.path.exists(repo_path):
        print(f"WARNING: The path '{repo_path}' does not exist!")
    
    # Add to the TOP of sys.path to ensure priority
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        
    return repo_path

def cleanup_conflicting_modules(modules_to_clear=['graph', 'graph.tools', 'model']):
    for mod in list(sys.modules.keys()):
        for target in modules_to_clear:
            if mod == target or mod.startswith(target + '.'):
                if mod in sys.modules:
                    del sys.modules[mod]

# --- Shared COCO 17 Definitions ---
def get_groups_coco17_0based():
    groups = []
    groups.append([11, 12])                  
    groups.append([5, 6, 13, 14])            
    groups.append([0, 7, 8, 15, 16])         
    groups.append([1, 2, 3, 4, 9, 10])       
    return groups

def get_groups_coco17_1based(dataset='Tennis', CoM=None):
    groups0 = get_groups_coco17_0based()
    groups1 = [[idx + 1 for idx in group] for group in groups0]
    return groups1

def get_edgeset_coco17_hierarchical():
    groups = get_groups_coco17_0based()
    identity, forward_hierarchy, reverse_hierarchy = [], [], []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(k, k) for k in self_link]
        identity.append(self_link)
        
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)
        
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
    num_node = 17
    self_link = [(i, i) for i in range(num_node)]
    # Inward: Leaf -> Center (Hip)
    # COCO 17 Topology
    inward = [
        (9, 7), (7, 5), (5, 11),  # L.Arm -> L.Hip
        (10, 8), (8, 6), (6, 12), # R.Arm -> R.Hip
        (15, 13), (13, 11),       # L.Leg -> L.Hip
        (16, 14), (14, 12),       # R.Leg -> R.Hip
        (11, 12),                 # L.Hip -> R.Hip (Connect Core)
        (0, 5), (0, 6),           # Nose -> Shoulders
        (1, 0), (2, 0),           # Eyes -> Nose
        (3, 1), (4, 2)            # Ears -> Eyes
    ]
    outward = [(j, i) for (i, j) in inward]
    return self_link, inward, outward

# --- HD-GCN ---

class GraphHD:
    def __init__(self, CoM=None, labeling_mode='spatial'):
        self.num_node = 17
        self.CoM = CoM if CoM is not None else 11 
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        # Imports must be local to avoid conflict before cleanup
        from graph import tools as graph_tools
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
             edges = get_edgeset_coco17_hierarchical()
             A = graph_tools.get_hierarchical_graph(self.num_node, edges)
        else:
            raise ValueError(f"Labeling mode {labeling_mode} not supported.")
        return A, self.CoM

class HDGCN_Tennis(nn.Module):
    def __init__(self, num_classes, in_channels=3, drop_out=0, **kwargs):
        super(HDGCN_Tennis, self).__init__()
        
        # 1. SETUP PATHS
        # Point to the folder that CONTAINS the 'model' and 'graph' folders
        setup_import_env(os.path.join(constants.PATH_FOR_AUXILIARY_DATASETS, 'HD-GCN-main'))

        # 2. CLEANUP
        cleanup_conflicting_modules()

        # 2. Import Official Model
        try:
            import model.HDGCN as OfficialHDGCN
            # Monkey patch
            OfficialHDGCN.get_groups = get_groups_coco17_1based
        except ImportError:
             # Retry with cleanup?
             cleanup_conflicting_modules()
             import model.HDGCN as OfficialHDGCN
             OfficialHDGCN.get_groups = get_groups_coco17_1based

        graph_args = {'CoM': 11, 'labeling_mode': 'spatial'}
        
        self.in_channels = in_channels
        self.model = OfficialHDGCN.Model(
            num_class=num_classes,
            num_point=17,
            num_person=1,
            graph='action_recognition.model.GraphHD',
            graph_args=graph_args,
            in_channels=in_channels,
            drop_out=drop_out
        )
        
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
            
        # Slicing input to match model channels
        if x.shape[1] > self.in_channels:
            x = x[:, :self.in_channels, :, :, :]
            
        return self.model(x)

# --- CTR-GCN ---
'''
class GraphCTR:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 17
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        from graph import tools as graph_tools
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            self_link, inward, outward = get_edgeset_coco17_spatial()
            A = graph_tools.get_spatial_graph(self.num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Labeling mode {labeling_mode} not supported.")
        return A

class CTRGCN_Tennis(nn.Module):
    def __init__(self, num_classes, in_channels=3, drop_out=0, **kwargs):
        super(CTRGCN_Tennis, self).__init__()
        
        # 1. Setup Environment
        
        setup_import_env(os.path.join(constants.PATH_FOR_AUXILIARY_DATASETS, 'CTR-GCN-main'))
        
        # 2. Import Official Model
        # We MUST clear at least 'graph' because GraphCTR uses 'graph.tools.get_spatial_graph' 
        # which has different signature in HDGCN.
        cleanup_conflicting_modules()
        
        import model.ctrgcn as OfficialCTRGCN
        
        graph_args = {'labeling_mode': 'spatial'}
        
        self.in_channels = in_channels
        self.model = OfficialCTRGCN.Model(
            num_class=num_classes,
            num_point=17,
            num_person=1,
            graph='action_recognition.model.GraphCTR',
            graph_args=graph_args,
            in_channels=in_channels,
            drop_out=drop_out,
            adaptive=True # Default
        )

    def forward(self, x):
        # CTR-GCN handles (N, C, T, V, M)
        if x.dim() == 4:
            x = x.unsqueeze(-1)
            
        # Slicing input to match model channels
        if x.shape[1] > self.in_channels:
            x = x[:, :self.in_channels, :, :, :]
            
        return self.model(x)
'''