from typing import Optional
from tqdm import tqdm

import numpy as np
import torch

from Oinfo.dataset import KNearestNeighborDataset
from torch.utils.data import DataLoader


def pairwise_var_diffs(X: np.ndarray):

    # X = (n_samples, n_variables)

    # (n_samples, n_samples, n_variables)
    return np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0)  


def entropy(X: torch.Tensor, k: int=3, base: int=2):

    # X[:, i, j, k] = sample_{i,k} - sample_{j,k} for each batch element (avoided for simplicity)

    # X = (batch_size, n_samples, n_samples, n_variables)
    _, n_samples, _, n_variables = X.shape
    
    # (batch_size, n_samples, n_samples)
    X_distances = torch.linalg.vector_norm(X, ord=2, dim=3)

    # (batch_size, n_samples, n_samples)
    X_distances, _ = torch.sort(X_distances, descending=False, dim=2)

    # (batch_size, n_samples, )
    knn_distances = X_distances[:, :, k+1]

    # TODO: check if is inportant to convert n_samples and k to tensors or if n_variables should be a tensor as well
    # TODO: export torch.log(torch.tensor(2.0)) as a constant
    const = (
        torch.digamma(torch.tensor(n_samples, dtype=torch.float32)) -
        torch.digamma(torch.tensor(k, dtype=torch.float32)) + 
        n_variables * torch.log(torch.tensor(2.0))
    )
    
    # (batch_size, )
    return (const + n_variables * torch.log(knn_distances.mean(dim=1))) / torch.log(torch.tensor(base))


def single_entropies(X: torch.Tensor):

    # X[:, i, j, k] = sample_{i,k} - sample_{j,k} for each batch element (avoided for simplicity)

    # X = (batch_size, n_samples, n_samples, n_variables)
    batch_size, n_samples, _, n_variables = X.shape

    tqdm.write('X.shape' + str(X.shape))

    # (batch_size, n_variables, n_samples, n_samples)
    X_permuted = X.permute(0, 3, 1, 2)

    tqdm.write('X_permuted.shape' + str(X_permuted.shape))

    # (batch_size * n_variables, n_samples, n_samples)
    X_reshaped = X_permuted.reshape(batch_size * n_variables, n_samples, n_samples)

    # (batch_size * n_variables, n_samples, n_samples, 1)
    X_reshaped = X_reshaped.unsqueeze(-1)

    # (batch_size * n_variables,)
    entropies = entropy(X_reshaped)

    # (batch_size, n_variables)
    return entropies.view(batch_size, n_variables)


def single_exclusion_entropies(X: torch.Tensor):

    # X[:, i, j, k] = sample_{i,k} - sample_{j,k} for each batch element (avoided for simplicity)

    # X = (batch_size, n_samples, n_samples, n_variables)
    n_variables = X.shape[3]

    # (n_variables, n_variables)
    masks = torch.ones((n_variables, n_variables), dtype=bool)
    masks[torch.arange(n_variables), torch.arange(n_variables)] = False
 
    # NOTE: The stacked loop could be improuved using advanced indexing to 
    # transform the X to shape (batch_size * n_variables, n_samples, n_sample, n_variables-1)
    # then compute the entropy in a single pass and then reshape to get (batch_size, n_features).
    # 
    # The drawback with that is that is much more memory consuming and not sure its a good optimization
    # and for sure is not that good if not run in GPU. Maybe is a good optimization to do to use
    # in GPU with a big RAM and select the metod on runtime depending on where is run

    # (batch_size, n_features)
    return torch.stack([
        entropy(X[:, :, :, mask])
        for mask in masks
    ], dim=1)


def o_information(X: torch.Tensor):
    """
    Calculate the O-information for a given dataset.

    Args:
        X (torch.Tensor): A batched 3D array representing the pairwise variabled
        difference. With shape (batch_size, n_samples, n_samples, n_variables).
        X[*, i, j, k] = sample_{i,k} - sample_{j,k}

    Returns:
        torch.Tensor: Computed O-information for all the batch.
    """

    # 

    n_variables = X.shape[3]

    # TODO: The single variable entropies are always the same.
    # This could be calculated once at the begining and then accessed here.

    # (batch_size, )
    joint_entropy = entropy(X)
    
    # (batch_size, n_features)
    all_individual_entropies = single_entropies(X)

    # (batch_size, n_features)
    all_residual_entropies = single_exclusion_entropies(X)

    # (batch_size, )
    return (n_variables - 2) * joint_entropy + (all_individual_entropies - all_residual_entropies).sum(dim=1)


def multi_order_meas_knn(X: np.ndarray, min_n: int=2, max_n: Optional[int]=None, batch_size: int=1000000):
    """    
    X (np.ndarray): The system data as a 2D array with shape (n_samples, n_variables)    
    
    """

    # make device cpu if not cuda available or cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    T, N = np.shape(X)
    n = N if max_n is None else max_n

    assert n < N, "max_n must be lower than len(elids). max_n >= len(elids))"
    assert min_n <= n, "min_n must be lower or equal than max_n. min_n > max_n"

    X_diffs = pairwise_var_diffs(X)

    # To compute using pytorch, we need to compute each order separately
    for order in range(min_n, n+1):

        dataset = KNearestNeighborDataset(X_diffs, N, order)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        for (partition_idxs, sub_distances) in tqdm(dataloader, total=(len(dataset) // batch_size)):

            sub_distances.to(device)

            o_info = o_information(sub_distances)

            return {
                'partition_idxs': partition_idxs,
                'order': order,
                'o_info': o_info.cpu().numpy()
            }
