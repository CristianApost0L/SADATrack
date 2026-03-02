import torch
import torch.nn as nn
import geoopt

class PoincareLinear(nn.Module):
    """
    Linear layer defined on the Poincaré ball model.
    Transforms points on the manifold via Möbius matrix multiplication,
    and optionally adds a bias via Möbius addition.
    """
    def __init__(self, in_features, out_features, manifold, bias=True, c=1.0):
        super(PoincareLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.c = torch.tensor([c], dtype=torch.float32)

        # Weight matrix in Euclidean space
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # Bias vector on the manifold
            self.bias = geoopt.ManifoldParameter(
                torch.zeros(out_features), manifold=self.manifold
            )
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        if self.bias is not None:
            # Initialize bias to origin on the Poincaré ball
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x):
        """
        x: Input tensor on the Poincaré ball [N, in_features]
        Returns: Tensor on the Poincaré ball [N, out_features]
        """
        # Möbius matrix multiplication (x @ W^T)
        res = self.manifold.mobius_matvec(self.weight, x)
        
        # Möbius addition of bias
        if self.bias is not None:
            res = self.manifold.mobius_add(res, self.bias)
            
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class HyperbolicMLR(nn.Module):
    """
    Hyperbolic Multinomial Logistic Regression.
    Maps Euclidean features to the Poincaré ball, applies a series of PoincareLinear layers (if hidden given), 
    and computes distances to hyperplanes for classification.
    """
    def __init__(self, in_features, num_classes, c=1.0):
        """
        Args:
            in_features: Dimension of Euclidean input features
            num_classes: Number of target classes
            c: Curvature of the Poincaré ball
        """
        super(HyperbolicMLR, self).__init__()
        
        # Define the Poincaré ball manifold with fixed curvature
        self.c = torch.tensor([c], dtype=torch.float32)
        # Using learnable curvature (optional but good for embeddings)
        # Here we fix it for stability during initial tests
        self.manifold = geoopt.PoincareBall(c=c)

        # For Multi-class Logistic Regression in hyperbolic space, 
        # we learn margin hyperplanes defined by a normal vector (p) and a scalar offset (a).
        # We need set of hyperplanes equal to num_classes
        
        # Hyperplane normal vectors (in Euclidean tangent space at origin)
        self.p_normals = nn.Parameter(torch.Tensor(num_classes, in_features))
        # Scalar offsets
        self.a_offsets = nn.Parameter(torch.zeros(num_classes))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p_normals)
        nn.init.zeros_(self.a_offsets)

    def forward(self, x, return_embeddings=False):
        """
        Args:
            x: Euclidean feature representations [N, in_features]
            return_embeddings: If True, also returns the projected Poincare embeddings
        Returns:
            logits: Output logits for classification [N, num_classes]
            (optional) z: Poincare embeddings [N, in_features]
        """
        # 1. Project Euclidean features to Poincaré ball via exponential map at origin
        # Make sure x is close to 0 to avoid numerical instability when mapping
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(1e-15)
        # Scale to avoid going out of bounds (norm should be < 1/sqrt(c))
        # This is a safe approximation
        x_scaled = x / (x_norm + 1.0) 
        
        z = self.manifold.expmap0(x_scaled)
        
        # 2. Compute distances to margin hyperplanes
        # Distance definition in hyperbolic space for hyperplanes
        # d(z, H_a,p) = 1/sqrt(c) * arsinh( ... )
        # Using the formulation from "Hyperbolic Neural Networks" (Ganea et al., 2018)
        
        # We broadcast z [N, D] and p [C, D] -> [N, C, D]
        N = z.shape[0]
        C = self.p_normals.shape[0]
        
        # Project normals to tangent space at origin (they already are, just formatting)
        p = self.p_normals  # [C, D]
        a = self.a_offsets  # [C]
        
        distances = torch.zeros(N, C, device=x.device)
        
        for k in range(C):
            pk = p[k] # [D]
            ak = a[k] # scalar
            
            # Compute distance for class k
            # Denominator: (1 - c ||z||^2) ||pk||
            # inner takes (x, u, v) where x is the base point, u and v are tangent vectors
            # For z_norm_sq, we can just use the Euclidean norm scaled, as in the Poincare ball
            # the norm squared is simply the Euclidean norm squared
            z_norm_sq = torch.sum(z * z, dim=-1, keepdim=True)
            pk_norm = torch.norm(pk, p=2).clamp_min(1e-15)
            
            denom = (1.0 - self.c.to(x.device) * z_norm_sq) * pk_norm
            
            # Numerator: 2 sqrt(c) | <-pk \oplus z, pk> |
            # Using -pk as origin for the hyperplane
            minus_pk_plus_z = self.manifold.mobius_add(-pk.unsqueeze(0), z) # [N, D]
            inner_prod = torch.matmul(minus_pk_plus_z, pk.unsqueeze(1)) # [N, 1]
            
            num = 2.0 * torch.sqrt(self.c.to(x.device)) * inner_prod
            
            # Distance
            dist = (1.0 / torch.sqrt(self.c.to(x.device))) * torch.arcsinh(num / denom.clamp_min(1e-15))
            
            # Distance with offset (sign determines class membership, we use raw distance as logit)
            distances[:, k] = dist.squeeze(1) - ak
            
        # 3. Logits are simply the distances (or negative distances depending on formulation)
        # We want the distance to be POSITIVE when point is correctly classified
        if return_embeddings:
            return distances, z
        return distances
