import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat

# --- 1. GRAPH CONSTRUCTION ---
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_graph(num_node, edges):
    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    A = np.stack((I, Forward, Reverse))
    return A

def get_hierarchical_graph(num_node, edges):
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A

def get_groups_coco17():
    """
    Define the hierarchy for the 17-point skeleton (COCO).
    Level 0 (Root): Hips (Left Hip 11, Right Hip 12)
    Level 1: Shoulders (5,6) and Knees (13,14
    Level 2: Elbows (7,8), Ankles (15,16), Nose (0)
    Level 3: Wrists (9,10), Eyes/Ears (1,2,3,4)
    """
    groups = []
    groups.append([11, 12])                  # 0. Core
    groups.append([5, 6, 13, 14])            # 1. Torso/Legs base
    groups.append([0, 7, 8, 15, 16])         # 2. Mid Limbs / Head base
    groups.append([1, 2, 3, 4, 9, 10])       # 3. Extremities
    return groups

def get_edgeset_coco17():
    groups = get_groups_coco17()
    
    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

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

# --- 2. LAYERS & BLOCKS ---
def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class HDGCN_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(HDGCN_TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=False) # Bias must be False for HDGCN weights
        
        # HDGCN uses a separate parameter for bias
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.bn(x)
        return x

class HDGCN_MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilations=[1,2], residual=True):
        super(HDGCN_MultiScaleTemporalConv, self).__init__()
        assert out_channels % (len(dilations) + 2) == 0
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(dilations)

        # Branch 1: 1x1 -> BN -> ReLU -> TemporalConv
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                HDGCN_TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Branch 3: MaxPool -> 1x1
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Branch 4: 1x1 (Resizing)
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = HDGCN_TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        branch_outs = [b(x) for b in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out += self.residual(x)
        return out

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d): conv_init(m)
            elif isinstance(m, nn.BatchNorm2d): bn_init(m, 1)

    def forward(self, x, dim=4): 
        # x: N, C, T, V (if dim=4) or N, C, L (if dim=3 for AHA)
        if dim == 4:
            N, C, T, V = x.size()
            x = x.mean(dim=-2) # Pool over time -> N, C, V
        else:
            N, C, L = x.size()
            V = L # Just naming
            
        x_graph = self.get_graph_feature(x, self.k)
        x_out = self.conv(x_graph)
        x_out = x_out.max(dim=-1)[0] # Max pooling over k neighbors

        if dim == 4:
            x_out = repeat(x_out, 'n c v -> n c t v', t=T)
            
        return x_out
        
    def get_graph_feature(self, x, k):
        N, C, V = x.size()
        # KNN logic
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        idx = dist.topk(k=k, dim=-1)[1] # N, V, k

        idx_base = torch.arange(0, N, device=x.device).view(-1, 1, 1) * V
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x_flat = rearrange(x, 'n c v -> n v c')
        x_flat = rearrange(x_flat, 'n v c -> (n v) c')
        
        feature = x_flat[idx, :]
        feature = feature.view(N, V, k, C)
        
        x_expanded = repeat(x, 'n c v -> n v k c', k=k)
        
        feature = torch.cat((feature - x_expanded, x_expanded), dim=3) # N, V, k, 2C
        feature = rearrange(feature, 'n v k c -> n c v k')
        return feature

class AHA(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(AHA, self).__init__()
        self.num_layers = num_layers
        groups = get_groups_coco17()
        
        # Flatten groups logic to match layers
        # Layer 0 connects Group 0 and 1
        # Layer 1 connects Group 1 and 2 ...
        self.layers_indices = [groups[i] + groups[i+1] for i in range(len(groups)-1)]
        
        inter_channels = in_channels // 4
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = EdgeConv(inter_channels, inter_channels, k=3)
        self.aggregate = nn.Conv1d(inter_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: N, C, num_layers, T, V
        N, C, L, T, V = x.size()
        x_t = x.max(dim=-2)[0] # Pool T -> N, C, L, V
        x_t = self.conv_down(x_t) # N, C', L, V
        
        # Sampling relevant nodes per layer
        x_sampled = []
        for i in range(self.num_layers):
            # Indices for layer i
            indices = self.layers_indices[i]
            # Select only these nodes from V dimension
            # Note: The original code logic is a bit implicit. 
            # In HD-GCN, `out` is stacked over layers. x[:,:,i,:] corresponds to the output of layer i.
            # But layer i operates on a subset of nodes? 
            # Actually HD_Gconv outputs full V for each layer, but zeroed? 
            # Let's verify HD_Gconv. No, HD_Gconv produces 'y' which is concatenated results.
            # However, to keep it robust for this adaptation:
            s_t = x_t[:, :, i, indices].mean(dim=-1, keepdim=True) # Average features of the group
            x_sampled.append(s_t)
            
        x_sampled = torch.cat(x_sampled, dim=2) # N, C', L
        
        att = self.edge_conv(x_sampled, dim=3) # N, C', L
        att = self.aggregate(att).unsqueeze(-1).unsqueeze(-1) # N, C, L, 1, 1
        
        out = (x * self.sigmoid(att)).sum(dim=2) # Weighted sum over layers -> N, C, T, V
        return out

class HD_Gconv(nn.Module):
    def __init__(self, in_channels, out_channels, A, att=False):
        super(HD_Gconv, self).__init__()
        self.num_layers = A.shape[0] # Number of hierarchy levels
        self.num_subset = A.shape[1] # 3 (Identity, Forward, Reverse)
        self.att = att
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        
        inter_channels = out_channels // (self.num_subset + 1)
        
        self.conv_down = nn.ModuleList()
        self.conv = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.conv_down.append(nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            ))
            conv_d = nn.ModuleList()
            for j in range(self.num_subset):
                conv_d.append(nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
                    nn.BatchNorm2d(inter_channels)
                ))
            conv_d.append(EdgeConv(inter_channels, inter_channels, k=5))
            self.conv.append(conv_d)
            
        if self.att:
            self.aha = AHA(out_channels, num_layers=self.num_layers)
            
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A = self.PA
        out = []
        for i in range(self.num_layers):
            y = []
            x_down = self.conv_down[i](x)
            for j in range(self.num_subset):
                # Graph Conv: (N, C, T, V) x (V, V)
                z = torch.einsum('n c t u, v u -> n c t v', x_down, A[i, j])
                z = self.conv[i][j](z)
                y.append(z)
            # Edge Conv part
            y_edge = self.conv[i][-1](x_down)
            y.append(y_edge)
            
            y = torch.cat(y, dim=1) # Concatenate along channels
            out.append(y)
            
        out = torch.stack(out, dim=2) # N, C, Layers, T, V
        
        if self.att:
            out = self.aha(out)
        else:
            out = out.sum(dim=2)
            
        out = self.bn(out)
        out += self.down(x)
        return self.relu(out)

class HDGCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, att=True):
        super(HDGCN_Unit, self).__init__()
        # We reuse HD_Gconv from previous definition as it doesn't use TemporalConv directly
        # If HD_Gconv is missing, please re-run the HDGCN definition block.
        # Assuming HD_Gconv is available in namespace or we redefine it simply:
        self.gcn1 = HD_Gconv(in_channels, out_channels, A, att=att) 
        self.tcn1 = HDGCN_MultiScaleTemporalConv(out_channels, out_channels, stride=stride, residual=False)
        self.relu = nn.ReLU(inplace=True)
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # Residual uses the specific TemporalConv too
            self.residual = HDGCN_TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

# --- 3. MAIN MODEL ---
class HDGCN_Tennis(nn.Module):
    def __init__(self, num_classes, in_channels=3, drop_out=0):
        super(HDGCN_Tennis, self).__init__()
        
        # Load Graph
        edges = get_edgeset_coco17() # Make sure this function is available
        self.A = get_hierarchical_graph(17, edges) 
        
        self.data_bn = nn.BatchNorm1d(1 * in_channels * 17)
        base_channels = 64
        
        # Build Layers using HDGCN_Unit
        self.l1 = HDGCN_Unit(in_channels, base_channels, self.A, residual=False, att=False)
        self.l2 = HDGCN_Unit(base_channels, base_channels, self.A)
        self.l3 = HDGCN_Unit(base_channels, base_channels, self.A)
        self.l4 = HDGCN_Unit(base_channels, base_channels, self.A)
        self.l5 = HDGCN_Unit(base_channels, base_channels*2, self.A, stride=2)
        self.l6 = HDGCN_Unit(base_channels*2, base_channels*2, self.A)
        self.l7 = HDGCN_Unit(base_channels*2, base_channels*2, self.A)
        self.l8 = HDGCN_Unit(base_channels*2, base_channels*4, self.A, stride=2)
        self.l9 = HDGCN_Unit(base_channels*4, base_channels*4, self.A)
        self.l10 = HDGCN_Unit(base_channels*4, base_channels*4, self.A)
        
        self.fc = nn.Linear(base_channels*4, num_classes)
        
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
            
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        x = x.mean(dim=3).mean(dim=2)
        x = self.drop_out(x)
        return self.fc(x)