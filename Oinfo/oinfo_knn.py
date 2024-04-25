from typing import Optional
import numpy as np

import torch

from Oinfo.dataset import LinpartsDataset


def entropy_torch(x, k=3, base=2):
    """
    PyTorch-based k-nearest neighbor entropy estimator that computes entropy for each batch element independently.
    `x` should be a tensor of shape (batch_size, n_samples, n_features) = (BS, T, N).
    `k` is the number of nearest neighbors to consider.
    """
    
    assert k < x.shape[1], f"Set k smaller than n_samples. {k} >= {x.shape[1]}."

    x = x + 1e-10 * torch.rand_like(x)  # Adding small noise to break ties
    batch_size, n_samples, n_features = x.shape
    
    # Compute pairwise distances
    x = x.unsqueeze(2)  # shape: (batch_size, n_samples, 1, n_features)
    distances = torch.norm(x - x.transpose(1, 2), dim=3, p=2)  # shape: (batch_size, n_samples, n_samples)

    # Get the k-nearest neighbors distances
    knn_distances, _ = torch.topk(distances, k+1, largest=False, dim=2)
    knn_distances = knn_distances[:, :, k+1]  # get only the k-th distances for each element shape: (batch_size, n_samples)

    # Compute entropy
    const = torch.digamma(torch.tensor(n_samples, dtype=torch.float32)) - torch.digamma(torch.tensor(k, dtype=torch.float32)) + n_features * torch.log(torch.tensor(2.0))
    entropy_values = (const + n_features * torch.log(knn_distances.mean(dim=1))) / torch.log(torch.tensor(base))

    # (batch_size, )
    return entropy_values  # Return entropy for each batch element


def total_entropy(X):

    batch_size, n_features, n_samples = X.shape

    # (batch_size, n_features, n_samples, 1)
    X_reshaped = X.permute(0, 2, 1).unsqueeze(-1)

    # (batch_size * n_features, n_samples, 1)
    X_flattened = X_reshaped.reshape(batch_size * n_features, n_samples, 1)

    # (batch_size * n_features,)
    entropies = entropy_torch(X_flattened)

    # (batch_size, n_features)
    return entropies.view(batch_size, n_features)


def conditional_entropy_torch(X):

    N = X.shape[1]

    single_exclusions_mask = (np.ones((N, N)) - np.eye(N)).astype(bool)

    # (batch_size, n_features)
    return torch.stack([
        entropy_torch(X[:, :, idxs])
        for idxs in single_exclusions_mask
    ], dim=1)


def o_information(X: np.ndarray):
    """
    Calculate the O-information for a given dataset.

    Args:
        X (np.ndarray): A batched 2D array representing multivariate features (batch_size, T samples x order features).
        single_exclusions_mask (np.ndarray): A boolean mask 2D array to exclude one variable at a time (order x order).
        entropy_func (callable): Function to compute entropy of the dataset.

    Returns:
        float: Computed O-information for the dataset.
    """

    N = X.shape[1]

    # (batch_size, )
    joint_entropy = entropy_torch(X)
    
    # (batch_size, n_features)
    individual_entropies = total_entropy(X)

    # (batch_size, n_features)
    conditional_entropies = conditional_entropy_torch(X)

    # (batch_size, )
    return (N - 2) * joint_entropy + (individual_entropies - conditional_entropies).sum(dim=1)


def multi_order_meas(data: np.ndarray, min_n: int=2, max_n: Optional[int]=None, batch_size: int=1000000):
    """    
    data = T samples x N variables matrix    
    
    """

    # make device cpu if not cuda available or cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    T, N = np.shape(data)
    n = N if max_n is None else max_n

    assert n < N, "max_n must be lower than len(elids). max_n >= len(elids))"
    assert min_n <= n, "min_n must be lower or equal than max_n. min_n > max_n"


    # Ver si puedo precacular la matriz de distancias y subindexar eso como lo hacemos
    # con la matriz de covarianzas covmat para optimizar. Esto es como definir la topologia entera
    # y uego calcular sobre ella
    
    # To compute using pytorch, we need to compute each order separately
    for order in range(min_n, n+1):

        dataset = LinpartsDataset(covmat, order)
        chunked_linparts_generator = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        pbar = tqdm(enumerate(chunked_linparts_generator), total=(len(dataset) // batch_size))
        for i, (linparts, psizes, all_covmats) in pbar:

            pbar.set_description(f'Processing chunk {order} - {i}: computing nplets')
            nplets_tc, nplets_dtc, nplets_o, nplets_s = _get_tc_dtc_from_batched_covmat(
                all_covmats, order,
                allmin1, bc1, bcN, bcNmin1,
                device
            )

            return {
                'linparts': linparts,
                'psizes': psizes,
                'nplets_tc': nplets_tc,
                'nplets_dtc': nplets_dtc,
                'nplets_o': nplets_o,
                'nplets_s': nplets_s
            }