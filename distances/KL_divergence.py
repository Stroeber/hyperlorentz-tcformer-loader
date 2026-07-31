import torch

class KLDivergenceDistance:
    def __init__(self, eps=1e-6):
        """
        Computes the Symmetric KL-Divergence between the multivariate normal 
        distributions of two subjects.
        
        eps: Small regularization value added to the covariance diagonal 
             to guarantee invertibility.
        """
        self.eps = eps

    def _estimate_gaussian_params(self, data):
        """
        Shapes data from (num_trials, channels, time_steps) into 
        (num_trials * time_steps, channels) to treat every time point 
        as a spatial observation sample.
        """
        data = data.squeeze(1)
        num_trials, channels, time_steps = data.size()
        
        # Permute and reshape to (Samples, Channels)
        samples = data.permute(0, 2, 1).reshape(-1, channels)
        
        # Compute mean vector over channels
        mu = samples.mean(dim=0)
        
        # Compute covariance matrix
        centered = samples - mu
        cov = (centered.T @ centered) / (samples.size(0) - 1)
        
        # Regularization to prevent singular matrix errors during inversion
        cov += self.eps * torch.eye(channels, device=data.device)
        
        return mu, cov

    def _kl_mvn(self, mu_0, cov_0, mu_1, cov_1):
        """
        Computes the one-way KL-Divergence: KL(N_0 || N_1)
        """
        d = mu_0.size(0)
        
        # Invert the covariance of the second distribution
        inv_cov_1 = torch.linalg.inv(cov_1)
        
        # 1. Trace Term: tr(Sigma_1^{-1} * Sigma_0)
        trace_term = torch.trace(inv_cov_1 @ cov_0)
        
        # 2. Mahalanobis Distance Term: (mu_1 - mu_0)^T * Sigma_1^{-1} * (mu_1 - mu_0)
        diff = mu_1 - mu_0
        mahalanobis_term = diff @ inv_cov_1 @ diff
        
        # 3. Log-Determinant Term: log(det(Sigma_1) / det(Sigma_0))
        # Using sign/logdet for extreme numerical stability
        _, logdet_0 = torch.linalg.slogdet(cov_0)
        _, logdet_1 = torch.linalg.slogdet(cov_1)
        logdet_term = logdet_1 - logdet_0
        
        # Full closed-form formulation
        kl = 0.5 * (trace_term + mahalanobis_term - d + logdet_term)
        return kl

    def __call__(self, data_A, data_B):
        """
        Calculates the symmetric KL distance wrapper for the matrix function.
        """
        # Step 1: Estimate distributions for both subjects
        mu_A, cov_A = self._estimate_gaussian_params(data_A)
        mu_B, cov_B = self._estimate_gaussian_params(data_B)
        
        # Step 2: Compute both directions
        kl_AB = self._kl_mvn(mu_A, cov_A, mu_B, cov_B)
        kl_BA = self._kl_mvn(mu_B, cov_B, mu_A, cov_A)
        
        # Step 3: Symmetrize
        symmetric_kl = 0.5 * (kl_AB + kl_BA)
        
        # Prevent trivial negative float rounding issues near zero
        return torch.clamp(symmetric_kl, min=0.0)