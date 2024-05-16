from typing import Optional

import scipy as sp
import numpy as np

from tqdm.autonotebook import tqdm

import pandas as pd
import torch

from .dataset import CovarianceDataset
from torch.utils.data import DataLoader


def _save_to_csv(output_path, nplet_tc, nplet_dtc, nplet_o, nplet_s, linparts, psizes, N, only_synergestic=False):

    # if only_synergestic; remove nplets with nplet_o >= 0
    if only_synergestic:
        print('Removing non synergetic values')
        to_keep = np.where(nplet_o < 0)[0]
        nplet_tc = nplet_tc[to_keep]
        nplet_dtc = nplet_dtc[to_keep]
        nplet_o = nplet_o[to_keep]
        nplet_s = nplet_s[to_keep]
        linparts = [linparts[i] for i in to_keep]
        psizes = psizes[to_keep]

    df_res = pd.DataFrame({
        'tc': nplet_tc,
        'dtc': nplet_dtc,
        'o': nplet_o,
        's': nplet_s,
        'psizes': psizes
    })

    # create a numpy matrix of with len(nplet_s) rows and N columns
    # each row is a boolean vector with True in the indexes of the nplet
    # and False in the rest
    nplet_matrix = np.zeros((len(nplet_s), N), dtype=bool)
    for i, nplet in tqdm(enumerate(linparts), total=len(linparts), leave=False, desc='Creating nplet matrix'):
        nplet_matrix[i, nplet] = True

    df_res = pd.concat([df_res, pd.DataFrame(nplet_matrix)], axis=1)

    df_res.to_csv(output_path, index=False)


def data2gaussian(X):
    """
    Transform the data into a Gaussian copula and compute the covariance matrix.
    
    Parameters:
    - X: A 2D numpy array of shape (T, N) where T is the number of samples and N is the number of variables.
    
    Returns:
    - X_gaussian: The data transformed into the Gaussian copula (same shape as the parameter input).
    - X_gaussian_covmat: The covariance matrix of the Gaussian copula transformed data.
    """

    T = np.size(X, axis=0)

    assert X.ndim == 2, f'data must be 2D but got {X.ndim}D data input'

    # Step 1 & 2: Rank the data and normalize the ranks
    sortid = np.argsort(X,axis=0) # sorting indices
    copdata = np.argsort(sortid,axis=0) # sorting sorting indices
    copdata = (copdata+1)/(T+1) # normalized indices in the [0,1] range 

    # Step 3: Apply the inverse CDF of the standard normal distribution
    X_gaussian = sp.special.ndtri(copdata) #uniform data to gaussian

    # Handle infinite values by setting them to 0 (optional and depends on use case)
    X_gaussian[np.isinf(X_gaussian)] = 0

    # Step 4: Compute the covariance matrix
    X_gaussian_covmat = np.cov(X_gaussian.T)

    return X_gaussian, X_gaussian_covmat


def _gaussian_entropy_bias_correction(N,T):
    """Computes the bias of the entropy estimator of a 
    N-dimensional multivariate gaussian with T sample"""
    psiterms = sp.special.psi((T - np.arange(1,N+1))/2)
    return (N*np.log(2/(T-1)) + np.sum(psiterms))/2


def _gaussian_entropy_estimation(cov_det, n_variables):
    return 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e).pow(n_variables) * cov_det)


def _all_min_1_ids(n_variables):
    return [np.setdiff1d(range(n_variables),x) for x in range(n_variables)]


def _get_tc_dtc_from_batched_covmat(covmat: torch.Tensor, allmin1, bc1: float, bcN: float, bcNmin1: float):

    # covmat is a batch of covariance matrices
    # |bz| x |N| x |N|

    n_variables = covmat.shape[2]

    # |bz|
    batch_det = torch.linalg.det(covmat)
    # |bz| x |N|
    single_var_dets = torch.diagonal(covmat, dim1=-2, dim2=-1)
    # |bz| x |N|
    single_exclusion_dets = torch.stack([torch.linalg.det(covmat[:,ids][:,:,ids]) for ids in allmin1], dim=1)

    # |bz|
    sys_ent = _gaussian_entropy_estimation(batch_det, n_variables) - bcN
    # TODO: The single variable entropies are always the same.
    # This could be calculated once at the begining and then accessed here.
    var_ents = _gaussian_entropy_estimation(single_var_dets, 1) - bc1
    # |bz| x |N|
    ent_min_one = _gaussian_entropy_estimation(single_exclusion_dets, n_variables-1) - bcNmin1
    # |bz| x |N|

    # |bz|
    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    # TODO: -inf - inf return NaN in pytorch. Check how should I handle this.
    # |bz|
    nplet_dtc = torch.sum(ent_min_one, dim=1) - (n_variables-1.0)*sys_ent

    # |bz|
    nplet_o = nplet_tc - nplet_dtc
    # |bz|
    nplet_s = nplet_tc + nplet_dtc

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def nplet_tc_dtc(X: np.ndarray):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T, N = np.shape(X)

    covmat = data2gaussian(X)[1]
    covmat = torch.Tensor(np.expand_dims(covmat, axis=0))
    covmat.to(device)

    allmin1 = _all_min_1_ids(N)
    bc1 = _gaussian_entropy_bias_correction(1,T)
    bcN = _gaussian_entropy_bias_correction(N,T)
    bcNmin1 = _gaussian_entropy_bias_correction(N-1,T)

    nplet_tc, nplet_dtc, nplet_o, nplet_s = _get_tc_dtc_from_batched_covmat(
        covmat, allmin1, bc1, bcN, bcNmin1
    )

    return (
        nplet_tc.cpu().numpy()[0],
        nplet_dtc.cpu().numpy()[0],
        nplet_o.cpu().numpy()[0],
        nplet_s.cpu().numpy()[0]
    )


def multi_order_meas_gc(X: np.ndarray, min_n: int=3, max_n: Optional[int]=None, batch_size: int=1000000):
    """    
    data = T samples x N variables matrix    
    
    """

    # make device cpu if not cuda available or cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    T, N = np.shape(X)
    n = N if max_n is None else max_n

    assert n < N, "max_n must be lower than len(elids). max_n >= len(elids))"
    assert min_n <= n, "min_n must be lower or equal than max_n. min_n > max_n"

    # Gaussian Copula of data
    covmat = data2gaussian(X)[1]

    # To compute using pytorch, we need to compute each order separately
    for order in range(min_n, n+1):

        # TODO: ver si puedo cambiar allmin1 por single_exclusion_mask
        allmin1 = _all_min_1_ids(order)
        bc1 = _gaussian_entropy_bias_correction(1,T)
        bcN = _gaussian_entropy_bias_correction(order,T)
        bcNmin1 = _gaussian_entropy_bias_correction(order-1,T)

        dataset = CovarianceDataset(covmat, N, order)
        chunked_linparts_generator = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        pbar = tqdm(enumerate(chunked_linparts_generator), total=(len(dataset) // batch_size), leave=False)
        for i, (partition_idxs, partition_covmat) in pbar:

            partition_covmat.to(device)

            pbar.set_description(f'Processing chunk {order} - {i}: computing nplets')
            nplets_tc, nplets_dtc, nplets_o, nplets_s = _get_tc_dtc_from_batched_covmat(
                partition_covmat, allmin1, bc1, bcN, bcNmin1
            )

            return {
                'partition_idxs': partition_idxs,
                'order': order,
                'nplets_tc': nplets_tc.cpu().numpy(),
                'nplets_dtc': nplets_dtc.cpu().numpy(),
                'nplets_o': nplets_o.cpu().numpy(),
                'nplets_s': nplets_s.cpu().numpy()
            }

'''

util function to save to csv

only_synergestic=False
output_path='./'

pbar.set_description(f'Saving chunk {i}')
_save_to_csv(
    os.path.join(output_path, f'nplets_{order}_{i}.csv'),
    nplets_tc, nplets_dtc,
    nplets_o, nplets_s,
    linparts, psizes,
    N, only_synergestic=only_synergestic
)
'''
