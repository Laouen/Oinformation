import numpy as np

def o_information(X: np.ndarray, entropy_func, single_exclusions_mask=None, individual_entropies=None):
    """
    Calculate the O-information for a given dataset.

    Args:
        X (np.ndarray): A 2D array representing multivariate features (n_samples, n_variables).
        entropy_func (callable): Function to compute entropy of the dataset of shape (n_samples, n_variables).
        single_exclusions_mask (np.ndarray): a matrix of all Trues but the diagonal with shape (n_variable, n_variables) to exclude one variable at a time. Optional, if None, the matrix is calculated
        individual_entropies (np.ndarray): a 1D vector of size n_variables with the individual entropies of each variable in X. individual_entropies[i] is equal to entropy_func(X[:,i]). Optional, if None, the individual entropies calculated

    Returns:
        float: Computed O-information for the dataset.
    """

    N = X.shape[1]

    if single_exclusions_mask is None:
        single_exclusions_mask = (np.ones((N, N)) - np.eye(N)).astype(bool)

    # H(X)
    system_entropy = entropy_func(X)

    # H(X1), ..., H(Xn)
    if individual_entropies is None:
        individual_entropies = np.array([entropy_func(X[:,[idx]]) for idx in range(N)])

    # H(X-1), ..., H(X-n)
    single_exclusion_entropies = np.array([entropy_func(X[:,idxs]) for idxs in single_exclusions_mask])

    return (N - 2) * system_entropy + (individual_entropies - single_exclusion_entropies).sum()

