import torch
import torch.nn as nn
from hyperbolic_lib.lib import geoopt


def minkowski_metric(n, *, device=None, dtype=None):
    """η = diag([-1, 1, ..., 1])"""
    eta = torch.eye(n, device=device, dtype=dtype)
    eta[0, 0] = -1.0
    return eta


class LorentzRotationBoost(nn.Module):
    """
    A learnable Lorentz transform L ∈ SO^+(1, d) parameterized as
    L = (Boost in direction u with rapidity φ) ∘ (Spatial rotation R)
    or the reverse order. Works on row-vectors x ∈ R^{N} with N = 1 + d.

    - Rotation R ∈ O(d) is stored as a Stiefel (orthonormal) parameter via geoopt.
      (Optionally enforce det(R)=+1 at runtime.)
    - Boost is parameterized by unit direction u ∈ S^{d-1} (geoopt Sphere) and rapidity φ ∈ R.

    Args:
        n (int): ambient Minkowski dimension (time + space), i.e., N=1+d.
        order (str): 'RB' for Boost ∘ Rotation, 'BR' for Rotation ∘ Boost.
        ensure_proper (bool): if True, enforce det(R)>0 by flipping last column when needed.

    Forward:
        x' = x @ L^T   (since x is a batch of row vectors)
    """
    def __init__(self, n: int, order: str = "RB", ensure_proper: bool = True):
        super().__init__()
        assert n >= 2, "Need at least time + 1 spatial dimension"
        assert order in {"RB", "BR"}
        self.n = n
        self.d = n - 1
        self.order = order
        self.ensure_proper = ensure_proper

        # --- geoopt manifolds ---
        self.sphere = geoopt.manifolds.sphere.Sphere()
        self.stiefel = geoopt.manifolds.Stiefel()

        # --- Rotation (spatial, d×d) on Stiefel manifold ---
        R0 = torch.eye(self.d)
        self.R = geoopt.ManifoldParameter(R0, manifold=self.stiefel)

        # --- Boost direction (unit vector in R^d) on Sphere S^{d-1} ---
        u0 = torch.randn(self.d)
        u0 = u0 / (u0.norm() + 1e-9)
        self.u = geoopt.ManifoldParameter(u0, manifold=self.sphere)

        # --- Rapidity φ (scalar, unrestricted) ---
        self.phi = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _block_from_spatial(R_spatial: torch.Tensor) -> torch.Tensor:
        """Embed d×d spatial rotation into (1+d)×(1+d) as diag(1, R)."""
        d = R_spatial.shape[-1]
        n = d + 1
        I = torch.eye(n, device=R_spatial.device, dtype=R_spatial.dtype)
        M = I.clone()
        M[1:, 1:] = R_spatial
        return M

    def _rotation_matrix(self) -> torch.Tensor:
        Rsp = self.R
        if self.ensure_proper:
            # Ensure det(R)>0 (avoid reflections). Cheap and effective.
            # (Non-smooth when the sign flips; usually rare in practice.)
            if torch.det(Rsp).detach() < 0:
                Rsp = Rsp.clone()
                Rsp[:, -1] = -Rsp[:, -1]
        return self._block_from_spatial(Rsp)

    def _boost_matrix(self) -> torch.Tensor:
        """
        Standard boost with time-first convention:
            t' = γ (t - β n·x)
            x' = x + (γ-1)(n·x) n - γβ t n
        where β = tanh(φ), γ = cosh(φ), γβ = sinh(φ), and n is a unit direction.
        """
        u = self.u
        phi = self.phi
        d = self.d
        device, dtype = u.device, u.dtype

        ch = torch.cosh(phi)
        sh = torch.sinh(phi)

        I_d = torch.eye(d, device=device, dtype=dtype)
        uuT = torch.outer(u, u)

        # Assemble the (1+d)x(1+d) matrix
        # [ [  ch,      -sh u^T ],
        #   [ -sh u,  I_d + (ch-1) uu^T ] ]
        top_left = ch.view(1, 1)
        top_right = (-sh * u).view(1, d)
        bottom_left = (-sh * u).view(d, 1)
        bottom_right = I_d + (ch - 1.0) * uuT

        B = torch.cat(
            [torch.cat([top_left, top_right], dim=1),
             torch.cat([bottom_left, bottom_right], dim=1)],
            dim=0
        )
        return B

    def lorentz_matrix(self) -> torch.Tensor:
        """Return full (1+d)×(1+d) Lorentz matrix composed as requested."""
        Rot = self._rotation_matrix()
        Boo = self._boost_matrix()
        return Boo @ Rot if self.order == "RB" else Rot @ Boo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N) with N = 1 + d, time coordinate first.
        Returns: x' = x @ L^T  (same shape)
        """
        L = self.lorentz_matrix()
        return x @ L.T

    def lorentz_error(self) -> float:
        """
        Max |L^T η L - η|∞ to monitor numerical exactness of constraints.
        Should be ~1e-6 or smaller in fp32.
        """
        L = self.lorentz_matrix()
        eta = minkowski_metric(self.n, device=L.device, dtype=L.dtype)
        err = (L.T @ eta @ L - eta).abs().max().item()
        return err


class FastLorentzRotationBoost(nn.Module):
    """
    Efficient Lorentz linear layer for x in R^{B×(1+d)} (time-first).
    Applies either:
        order="RB":  x -> R (spatial) then Boost(u, phi)
        order="BR":  x -> Boost(u, phi) then R (spatial)

    Key efficiency trick:
      - No (1+d)x(1+d) assembly/matmul.
      - Boost is rank-1: uses u·x and a few axpy ops.

    Parameters (geoopt-constrained):
      R  : d×d Stiefel (orthonormal) for spatial rotation
      u  : unit vector on S^{d-1} for boost direction
      phi: scalar rapidity
    """
    def __init__(self, n: int, order: str = "RB", ensure_proper_init: bool = True):
        super().__init__()
        assert n >= 2, "Need at least time + 1 spatial dimension"
        assert order in {"RB", "BR"}
        self.n = n
        self.d = n - 1
        self.order = order

        # geoopt manifolds
        self.stiefel = geoopt.manifolds.stiefel.Stiefel()
        self.sphere = geoopt.manifolds.sphere.Sphere()

        # Rotation (spatial) on Stiefel
        R0 = torch.eye(self.d)
        self.R = geoopt.ManifoldParameter(R0, manifold=self.stiefel)

        # Boost direction on Sphere S^{d-1}
        u0 = torch.randn(self.d)
        u0 = u0 / (u0.norm() + 1e-9)
        self.u = geoopt.ManifoldParameter(u0, manifold=self.sphere)

        # Rapidity (free Euclidean scalar)
        self.phi = nn.Parameter(torch.zeros(()))

        if ensure_proper_init:
            # Make det(R) positive once at init (cheap, not done each forward)
            with torch.no_grad():
                if torch.det(self.R).item() < 0:
                    self.R[:, -1] *= -1.0

    @staticmethod
    def _boost(t, X, u, phi):
        """
        Boost with time-first convention:
          t' = ch * t - sh * (u·X)
          X' = X + (ch-1)*(u·X) * u - sh * t * u
        Shapes:
          t: (B, 1), X: (B, d), u: (d,), phi: scalar
        """
        ch = torch.cosh(phi)
        sh = torch.sinh(phi)

        # s = u·X -> (B, 1)
        s = X @ u.view(-1, 1)

        # t': (B,1)
        t_new = ch * t - sh * s

        # X': (B,d)
        # X + (ch-1)*s*u - sh*t*u    (broadcasts)
        X_new = X + (ch - 1.0) * s * u.view(1, -1) - sh * t * u.view(1, -1)
        return t_new, X_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1+d) with time coordinate first.
        Returns x' with the same shape.
        """
        t = x[:, :1]         # (B,1)
        X = x[:, 1:]         # (B,d)

        if self.order == "RB":
            # Rotate then Boost
            X = X @ self.R.T
            t, X = self._boost(t, X, self.u, self.phi)
        else:
            # Boost then Rotate
            t, X = self._boost(t, X, self.u, self.phi)
            X = X @ self.R.T

        return torch.cat([t, X], dim=1)

    # Optional: exact matrix if you need it (not used in forward, kept for checks)
    def lorentz_matrix(self) -> torch.Tensor:
        d = self.d
        I_d = torch.eye(d, device=self.R.device, dtype=self.R.dtype)
        ch, sh = torch.cosh(self.phi), torch.sinh(self.phi)
        u = self.u
        uuT = torch.outer(u, u)

        # Rotation block
        Rot = torch.eye(d + 1, device=self.R.device, dtype=self.R.dtype)
        Rot[1:, 1:] = self.R

        # Boost block
        B = torch.empty((d + 1, d + 1), device=self.R.device, dtype=self.R.dtype)
        B[0, 0] = ch
        B[0, 1:] = (-sh * u).view(-1)
        B[1:, 0] = (-sh * u).view(-1)
        B[1:, 1:] = I_d + (ch - 1.0) * uuT

        return B @ Rot if self.order == "RB" else Rot @ B

    def lorentz_error(self) -> float:
        """Max |L^T η L - η|∞ to monitor constraint accuracy."""
        L = self.lorentz_matrix()
        eta = torch.eye(self.n, device=L.device, dtype=L.dtype)
        eta[0, 0] = -1.0
        return (L.T @ eta @ L - eta).abs().max().item()
