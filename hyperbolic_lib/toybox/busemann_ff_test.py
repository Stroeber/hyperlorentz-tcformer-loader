import torch
import torch.nn.functional as F
import torch.nn as nn
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from hyperbolic_lib.lib.geoopt import ManifoldParameter


# Use Lorentz manifold from geoopt (handles curvature automatically)
class HorosphereFC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, k: float = 1.0):
        """
        input_dim: Dimension d of input hyperboloid (H^d)
        output_dim: Dimension m of output hyperboloid (H^m)
        k: Curvature parameter (k > 0)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create Lorentz manifold for input space
        self.manifold = CustomLorentz(k=k)
        self.k = self.manifold.k

        # Parameters: p_j ∈ H^d, ξ_j ∈ tangent space at origin (lightlike)
        self.p = ManifoldParameter(
            self.manifold.projx(torch.randn(output_dim-1, input_dim + 1)),
            manifold=self.manifold
        )

        # ξ direction parameters (in tangent space at origin)
        self.xi_dir = nn.Parameter(torch.randn(output_dim-1, input_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x ∈ H_k^d (shape: [batch_size, d+1])
        Output: y ∈ H_k^m (shape: [batch_size, m+1])
        """
        # Normalize ξ directions to unit norm (in Euclidean sense)
        z = self.xi_dir / torch.norm(self.xi_dir, dim=1, keepdim=True)  # Unit norm
        xi = torch.cat([
            torch.norm(z, dim=1, keepdim=True),  # ‖v‖
            z
        ], dim=1)   # [num_planes, d+1]

        xi = self.manifold.projx(xi)

        log_inner = torch.log(
            -self.manifold.minkowski_dot(
                x.unsqueeze(-2),  # [batch_size, 1, in_dim+1]
                xi.unsqueeze(0)  # [1, out_dim, in_dim+1]
            )
        )  # [batch_size, out_dim]

        busemann_x = torch.sqrt(self.k) * log_inner
        busemann_p = torch.sqrt(self.k) * torch.log(
            -self.manifold.minkowski_dot(self.p, xi)
        )  # [out_dim]

        # [batch_size, out_dim]
        v = busemann_x - busemann_p.unsqueeze(0)

        v = torch.nn.functional.pad(v.squeeze(-1), (1, 0, 0, 0), mode='constant', value=0)
        v = self.manifold.expmap0(v)

        # Apply exponential map
        return v


# Example Usage
if __name__ == "__main__":
    import geoopt

    # Settings with curvature k=0.5
    d_in, d_out, k = 16, 32, 0.5
    batch_size = 4

    # Create layer
    fc = HyperboloidFC_geoopt(input_dim=d_in, output_dim=d_out, k=k)

    # Create input points on H_k^d using geoopt
    manifold_in = CustomLorentz(k=k)
    x_hyper = manifold_in.projx(torch.randn(batch_size, d_in + 1))

    # Forward pass
    y_hyper = fc(x_hyper)

    print(f"Curvature: -1/{k}")
    print("Input shape:", x_hyper.shape)
    print("Output shape:", y_hyper.shape)

    # Verify output is on hyperboloid
    manifold_out = CustomLorentz(k=k)
    print("Hyperbolicity check:",
          manifold_out.check_point_on_manifold(y_hyper))
