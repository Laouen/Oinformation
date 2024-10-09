import numpy as np
from scipy.special import gammaln, psi
import sys
import os

from npeet import entropy_estimators as ee

GCMI_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libraries')
sys.path.append(GCMI_dir)

from gcmi.python import gcmi

from thoi.measures.gaussian_copula import nplets_measures


########################## Estimator formulas #############################


def npeet_entropy(X:np.ndarray):
    return ee.entropy(X, base=np.e)


def gcmi_entropy(X:np.ndarray):
    return gcmi.ent_g(gcmi.copnorm(X.T))


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


######################## Gausian formulas #########################


def gaussian_entropy(covariance_matrix):
    n_variables = covariance_matrix.shape[0]
    cov_det = np.linalg.det(covariance_matrix) 
    return 0.5 * np.log((2 * np.pi * np.e)**n_variables * cov_det)


def o_information_gaussian(covariance_matrix):
    """
    Calculate the O-information for a Gaussian distribution given its mean and covariance matrix.

    Args:
        mean (np.ndarray): Mean vector of the Gaussian distribution.
        covariance_matrix (np.ndarray): Covariance matrix of the Gaussian distribution.

    Returns:
        float: Computed O-information for the Gaussian distribution.
    """

    N = covariance_matrix.shape[0]

    # H(X)
    system_entropy = gaussian_entropy(covariance_matrix)

    # H(X1), ..., H(Xn)
    individual_entropies = np.array([
        gaussian_entropy(np.array([[covariance_matrix[i, i]]])) for i in range(N)])

    # H(X-1), ..., H(X-n)
    single_exclusion_entropies = np.array([
        gaussian_entropy(np.delete(np.delete(covariance_matrix, i, axis=0), i, axis=1))
        for i in range(N)
    ])

    return (N - 2) * system_entropy + (individual_entropies - single_exclusion_entropies).sum()


########################### Uniform formulas ###################################


def calculate_volume(lower_bounds, upper_bounds):
    return np.prod(np.array(upper_bounds) - np.array(lower_bounds))


def uniform_entropy(lower_bounds, upper_bounds):
    volume = calculate_volume(lower_bounds, upper_bounds)
    return np.log(volume)


def o_information_uniform(lower_bounds, upper_bounds):
    """
    Calculate the O-information for a uniform distribution given its bounds.

    Args:
        lower_bounds (np.ndarray): Lower bounds of the uniform distribution.
        upper_bounds (np.ndarray): Upper bounds of the uniform distribution.

    Returns:
        float: Computed O-information for the uniform distribution.
    """

    N = len(lower_bounds)

    # H(X)
    system_entropy = uniform_entropy(lower_bounds, upper_bounds)

    # H(X1), ..., H(Xn)
    individual_entropies = np.array([
        uniform_entropy(np.array([lower_bounds[i]]), np.array([upper_bounds[i]]))
        for i in range(N)
    ])

    # H(X-1), ..., H(X-n)
    single_exclusion_entropies = np.array([
        uniform_entropy(lower_bounds[np.arange(N) != i], upper_bounds[np.arange(N) != i])
        for i in range(N)
    ])

    return (N - 2) * system_entropy + (individual_entropies - single_exclusion_entropies).sum()


############################ Dirichlet formulas #################################


def dirichlet_entropy(alpha):
    alpha_0 = np.sum(alpha)
    k = len(alpha)
    
    # Multivariate beta function B(alpha) = (prod(gamma(alpha_i))) / gamma(sum(alpha_i))
    B_alpha = np.exp(np.sum(gammaln(alpha)) - gammaln(alpha_0))
    
    # Entropy calculation
    entropy = np.log(B_alpha) + (alpha_0 - k) * psi(alpha_0) - np.sum((alpha - 1) * psi(alpha))
    return entropy


def o_information_dirichlet(alpha):
    """
    Calculate the O-information for a Dirichlet distribution given its concentration parameters.

    Args:
        alpha (np.ndarray): Concentration parameters of the Dirichlet distribution.

    Returns:
        float: Computed O-information for the Dirichlet distribution.
    """

    N = len(alpha)

    # H(X)
    system_entropy = dirichlet_entropy(alpha)

    # H(X1), ..., H(Xn)
    individual_entropies = np.array([
        dirichlet_entropy(np.array([alpha[i]]))
        for i in range(N)
    ])

    # H(X-1), ..., H(X-n)
    single_exclusion_entropies = np.array([
        dirichlet_entropy(alpha[np.arange(N) != i])
        for i in range(N)
    ])

    return (N - 2) * system_entropy + (individual_entropies - single_exclusion_entropies).sum()


############################# o information thoi ###############################


def o_information_thoi(X:np.ndarray):
    return nplets_measures(X)[0][2]