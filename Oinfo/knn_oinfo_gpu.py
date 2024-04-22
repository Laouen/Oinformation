import numpy as np

import torch
from torch import nn

# TODO: correctly finish to implement this 

def entropy_torch(x, k=3, base=2):
    """
    PyTorch-based k-nearest neighbor entropy estimator that computes entropy for each batch element independently.
    `x` should be a tensor of shape (batch_size, n_samples, n_features).
    `k` is the number of nearest neighbors to consider.
    """
    assert k <= x.shape[1] - 1, "Set k smaller than num. samples - 1"
    x = x + 1e-10 * torch.rand_like(x)  # Adding small noise to break ties
    batch_size, n_samples, n_features = x.shape
    
    # Compute pairwise distances
    x = x.unsqueeze(2)  # shape: (batch_size, n_samples, 1, n_features)
    distances = torch.norm(x - x.transpose(1, 2), dim=3, p=2)  # shape: (batch_size, n_samples, n_samples)

    # Get the k-nearest neighbors distances
    knn_distances, _ = torch.topk(distances, k+1, largest=False, dim=2)
    knn_distances = knn_distances[:, :, 1:]  # Exclude self-distance

    # Compute entropy
    const = torch.digamma(torch.tensor(n_samples, dtype=torch.float32)) - torch.digamma(torch.tensor(k, dtype=torch.float32)) + n_features * torch.log(torch.tensor(2.0))
    entropy_values = (const + n_features * torch.log(knn_distances.mean(dim=2))) / torch.log(torch.tensor(base))

    return entropy_values  # Return entropy for each batch element


A = [45, 37, 42, 35, 39]
B = [38, 31, 26, 28, 33]
C = [10, 15, 17, 21, 12]

data = np.array([A, B, C])

data.shape


# Example usage
T = 100
N = 5
for bz in range(10):
    x = torch.randn(bz, T, N)  # 10 batches, 100 samples each, 5 features per sample

    x_ent = entropy_torch(x)

    print('x.shape', x.shape)
    print('x_ent.shape', x_ent.shape)