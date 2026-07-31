import torch

class CovarianceDistance:
    def __init__(self, metric='log_euclidean'):
        """
        Interchangeable covariance distance metric.
        Supported metrics: 'frobenius', 'log_euclidean'
        """
        self.metric = metric.lower()
        if self.metric not in ['frobenius', 'log_euclidean']:
            raise ValueError("Metric must be either 'frobenius' or 'log_euclidean'")

    def _compute_average_covariance(self, data):
        """
        Computes the average channel covariance matrix across all trials.
        Input data shape: (num_trials, channels, time_steps)
        Output shape: (channels, channels)
        """
        data = data.squeeze(1)
        # Step 1: Center the data across the time dimension (dim=2)
        mean_over_time = data.mean(dim=2, keepdim=True)
        centered_data = data - mean_over_time
        
        # Step 2: Batch matrix multiply centered profiles: (channels, time) @ (time, channels)
        # Using torch.bmm requires transposing the second matrix
        time_steps = data.size(2)
        trial_covariances = torch.bmm(centered_data, centered_data.transpose(1, 2)) / (time_steps - 1)
        
        # Step 3: Average across all trials
        mean_covariance = trial_covariances.mean(dim=0)
        return mean_covariance



    def __call__(self, data_A, data_B):
        """
        Calculates distance between Subject A and Subject B covariance distributions.
        Shapes: (num_trials, channels, time_steps)
        """
        # Compute summary covariance matrices
        cov_A = self._compute_average_covariance(data_A)
        cov_B = self._compute_average_covariance(data_B)
        
        if self.metric == 'frobenius':
            # Straightforward matrix difference norm
            distance = torch.linalg.norm(cov_A - cov_B, ord='fro')
            
        elif self.metric == 'log_euclidean':
            # Map to tangent space via matrix logarithm, then compute Frobenius norm
            # log_A = torch.linalg.logm(cov_A)
            # log_B = torch.linalg.logm(cov_B)
            u, s, v = torch.linalg.svd(cov_A)
            log_A = torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s))), v)
            
            u, s, v = torch.linalg.svd(cov_B)
            log_B = torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s))), v)
            
            distance = torch.linalg.norm(log_A - log_B, ord='fro')
            
        return distance