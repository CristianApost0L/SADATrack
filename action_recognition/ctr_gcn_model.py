import sys
import os
import torch
import torch.nn as nn
import numpy as np
from constants import COCO_NUM_JOINTS, COCO_CENTER_OF_MASS

# --- UTILS FOR IMPORTING EXTERNAL REPO ---
def setup_import_env(repo_rel_path):
    """Adds the external repo to sys.path so we can import its modules"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust this path depending on where you put the cloned CTR-GCN-main folder
    repo_path = os.path.join(current_dir, '..', repo_rel_path) 
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path

def cleanup_conflicting_modules(modules_to_clear=['graph', 'graph.tools', 'model']):
    """CRITICAL: Clears sys.modules to prevent conflict between HD-GCN and CTR-GCN"""
    for mod in list(sys.modules.keys()):
        for target in modules_to_clear:
            if mod == target or mod.startswith(target + '.'):
                if mod in sys.modules:
                    del sys.modules[mod]

# --- GRAPH DEFINITION FOR CTR-GCN ---
def get_edgeset_coco17_spatial():
    """Defines the physical connections of the COCO-17 skeleton"""
    num_node = 17
    self_link = [(i, i) for i in range(num_node)]
    
    # Inward: Extremities -> Center (Hips)
    inward = [
        (9, 7), (7, 5), (5, 11),  # L.Wrist -> L.Elbow -> L.Shoulder -> L.Hip
        (10, 8), (8, 6), (6, 12), # R.Wrist -> R.Elbow -> R.Shoulder -> R.Hip
        (15, 13), (13, 11),       # L.Ankle -> L.Knee -> L.Hip
        (16, 14), (14, 12),       # R.Ankle -> R.Knee -> R.Hip
        (11, 12),                 # L.Hip <-> R.Hip
        (0, 5), (0, 6),           # Nose -> Shoulders
        (1, 0), (2, 0),           # Eyes -> Nose
        (3, 1), (4, 2)            # Ears -> Eyes
    ]
    outward = [(j, i) for (i, j) in inward]
    return self_link, inward, outward

class GraphCTR:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = COCO_NUM_JOINTS
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        # We import tools dynamically to use the ones from the CTR-GCN repo
        from graph import tools as graph_tools 
        if labeling_mode == 'spatial':
            self_link, inward, outward = get_edgeset_coco17_spatial()
            A = graph_tools.get_spatial_graph(self.num_node, self_link, inward, outward)
            return A
        raise ValueError("Only 'spatial' labeling supported")

# --- WRAPPER MODEL CLASS ---
class CTRGCN_Tennis(nn.Module):
    def __init__(self, num_classes, in_channels=3, drop_out=0.5, **kwargs):
        super(CTRGCN_Tennis, self).__init__()
        
        # 1. Setup the environment to see the external folder
        # Make sure the folder name matches what you cloned (e.g., 'CTR-GCN-main')
        setup_import_env('/kaggle/input/cv-auxiliary-repos/CTR-GCN-main') 

        # 2. Clean namespaces to avoid conflict with HD-GCN imports
        cleanup_conflicting_modules()

        # 3. Import the official model
        import model.ctrgcn as OfficialCTRGCN
        
        # 4. Initialize graph
        self.graph = GraphCTR(labeling_mode='spatial')
        A = self.graph.A

        # 5. Instantiate the model
        # Note: We hardcode person=1 and points=17 for your use case
        self.model = OfficialCTRGCN.Model(
            num_class=num_classes,
            num_point=COCO_NUM_JOINTS,
            num_person=1,
            graph='action_recognition.ctr_gcn_model.GraphCTR',
            graph_args={'labeling_mode': 'spatial'},
            in_channels=in_channels,
            drop_out=drop_out,
            adaptive=True # Enable the core feature of CTR-GCN
        )
        # Monkey-patch the graph into the model if needed, or pass A directly if the API differs.
        # The official CTR-GCN often builds the graph internally based on the graph_args string.
        # However, to ensure it uses OUR COCO graph, we can manually set it if the repo allows,
        # or ensure 'graph.tools' in the repo is patched. 
        # For now, relying on the 'graph' argument in constructor is safest if passing a string class path.
        # But since we imported OfficialCTRGCN, we use it directly.

    def forward(self, x):
        # Input x: (N, C, T, V) or (N, C, T, V, M)
        if x.dim() == 4:
            x = x.unsqueeze(-1) # Add person dimension -> (N, C, T, V, M)
        
        return self.model(x)