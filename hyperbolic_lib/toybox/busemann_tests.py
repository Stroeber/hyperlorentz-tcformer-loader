import torch
import geoopt
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz


def busemann_lorentz(x: torch.Tensor,
                     ideal_point: torch.Tensor,
                     *,
                     model: CustomLorentz):
    """
    Compute Busemann function in Lorentz model.

    Args:
        x: Points on the manifold (..., dim)
        ideal_point: Boundary point (light-like vector) (..., dim)
        model: Lorentz manifold instance

    Returns:
        Busemann function values at x (...,)
    """
    # Ensure inputs are on the manifold and boundary
    x = model.assert_check_point_on_manifold(x)
    # Verify <ideal_point, ideal_point>_L = 0 (light-like)
    inner = model.inner(None, ideal_point, ideal_point)
    if not torch.allclose(inner, torch.zeros_like(inner), atol=1e-6):
        raise ValueError("ideal_point must be light-like (||ξ||_L = 0)")

    # Compute <-x, ξ>_L and clamp for numerical stability
    inner_val = -model.inner(None, x, ideal_point)
    inner_val = inner_val.clamp(min=1e-8)  # avoid negative values
    return -inner_val.log()


def lorentz_point_to_hyperplane(
        x: torch.Tensor,
        p: torch.Tensor,
        ideal_point: torch.Tensor,
        *,
        manifold: CustomLorentz
):
    """
    Compute signed distance from point x to hyperplane defined by
    base point p and boundary direction ideal_point in Lorentz model.

    Hyperplane: { y ∈ H^n | B_ξ(⊖p ⊕ y) = 0 }

    Args:
        x: Query points (..., dim)
        p: Base point on hyperplane (..., dim)
        ideal_point: Boundary direction (light-like) (..., dim)
        manifold: Lorentz manifold instance

    Returns:
        Signed distances (...,)
    """
    # Compute ⊖p (inverse operation in symmetric space)
    # In Lorentz model: ⊖p = reflection of p through origin
    minus_p = p.clone()
    minus_p[..., 0] = -minus_p[..., 0]  # Negate time component

    # Compute ⊖p ⊕ x = group operation applied to x
    # Using parallel transport: g^{-1}*x where g moves origin to p
    transport_x = manifold.transp0(p, x)  # ⊖p ⊕ x ≈ transport from p to origin

    # Compute Busemann function at transported point
    B_val = busemann_lorentz(transport_x, ideal_point, model=manifold)

    # Compute Riemannian distance d(x,p)
    d_x_p = manifold.dist(x, p, keepdim=False)

    # Compute norm ||⊖p ⊕ x||_S = d(x,p) (Proposition 4.8)
    # In symmetric spaces: ||g^{-1}*x||_S = d(x,p)
    norm_transport = d_x_p

    # Signed distance formula (Definition 4.2)
    signed_distance = d_x_p * (B_val / norm_transport)
    return signed_distance


# ================================================
# Example Usage
# ================================================
if __name__ == "__main__":
    # Initialize Lorentz manifold (hyperboloid model)
    manifold = CustomLorentz(k=1.0)  # k=1 for hyperboloid curvature

    # Random points on the manifold
    p = manifold.random(10, 3)  # (10, 3) points: (batch, dim)
    x = manifold.random(10, 3)

    # Create ideal point (light-like vector on boundary)
    # Example: spatial direction [0, 1, 0] with time=1
    ideal_point = torch.zeros(10, 3)
    ideal_point[..., 0] = 1.0  # time component
    ideal_point[..., 1] = 1.0  # spatial component (||ξ||=1)
    # Normalize to light-like: <ξ,ξ>_L = -1^2 + 1^2 = 0
    ideal_point = manifold.projx(ideal_point)

    # Compute signed distances
    distances = lorentz_point_to_hyperplane(
        x=x,
        p=p,
        ideal_point=ideal_point,
        manifold=manifold
    )

    print("Signed distances to hyperplanes:")
    print(distances)
