#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:06:34 2021

@author: Ruben Herzog
@contributor: Laouen Belloli
"""

import os

import scipy as sp
from scipy import signal
import scipy.signal as sig
import numpy as np
import math

from tqdm.autonotebook import tqdm
from itertools import combinations, chain, islice
from multiprocessing import Pool, cpu_count

import pandas as pd

import torch

import time


TD_DTC_GLOBAL_DATA = {
    'covmat': None
}


def _save_to_csv(output_path, nplet_tc, nplet_dtc, nplet_o, nplet_s, linparts, psizes, N, only_synergestic=False):

    # if only_synergestic; remove nplets with nplet_o >= 0
    if only_synergestic:
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


def data2gaussian(data):
    """
    INPUT
    data = T samples x N variables matrix
    
    OUTPUT
    gaussian_data = T samples x N variables matrix with the gaussian copula
    transformed data
    covmat = N x N covariance matrix of guassian copula transformed data.
    """
    T = np.size(data,axis=0)
    sortid = np.argsort(data,axis=0) # sorting indices
    copdata = np.argsort(sortid,axis=0) # sorting sorting indices
    copdata = (copdata+1)/(T+1) # normalized indices in the [0,1] range 
    gaussian_data = sp.special.ndtri(copdata) #uniform data to gaussian
    gaussian_data[np.isinf(gaussian_data)] = 0
    covmat = gaussian_data.T @ gaussian_data /(T-1) #GC covmat
    
    return gaussian_data,covmat


def _gauss_ent_biascorr(N,T):
    """Computes the bias of the entropy estimator of a 
    N-dimensional multivariate gaussian with T sample"""
    psiterms = sp.special.psi((T - np.arange(1,N+1))/2)
    return (N*np.log(2/(T-1)) + np.sum(psiterms))/2


def _gauss_ent_est(x,y):
    return 0.5 * torch.log((2 * torch.pi * torch.exp(torch.tensor(1.))).pow(x) * y)


def _all_min_1_ids(N):
    return [np.setdiff1d(range(N),x) for x in range(N)]


def _get_tc_dtc_from_batched_covmat(covmat, N, T, device):

    # covmat is a batch of covariance matrices

    batch_covmat = torch.tensor(covmat).to(device)

    # Compute parameters for the batch, this assumes all the batch is from the same order N
    allmin1 = _all_min_1_ids(N)
    bc1 = _gauss_ent_biascorr(1,T)
    bcN = _gauss_ent_biascorr(N,T)
    bcNmin1 = _gauss_ent_biascorr(N-1,T)

    batch_detmv = torch.linalg.det(batch_covmat)
    # TODO: check if the following line is correct, the subindexing may need to consider a : for the batch dimension.
    # maybe it can be done with torch.einsum or in a single [:,ids,:,ids] or something
    batch_detmv_min_1 = torch.stack([torch.linalg.det(batch_covmat[:,ids][:,:,ids]) for ids in allmin1]).T
    batch_single_vars = torch.diagonal(batch_covmat, dim1=-2, dim2=-1)

    var_ents = _gauss_ent_est(1.0, batch_single_vars) - bc1
    sys_ent = _gauss_ent_est(N, batch_detmv) - bcN
    ent_min_one = _gauss_ent_est(N-1.0, batch_detmv_min_1) - bcNmin1

    nplet_tc = torch.sum(var_ents, dim=1) - sys_ent
    nplet_dtc = torch.sum(ent_min_one, dim=1) - (N-1.0)*sys_ent

    nplet_o = nplet_tc - nplet_dtc
    nplet_s = nplet_tc + nplet_dtc

    # bring from device and convert to numpy
    nplet_tc = nplet_tc.cpu().numpy()
    nplet_dtc = nplet_dtc.cpu().numpy()
    nplet_o = nplet_o.cpu().numpy()
    nplet_s = nplet_s.cpu().numpy()

    return nplet_tc, nplet_dtc, nplet_o, nplet_s


def _get_covmat(nplet):
    return TD_DTC_GLOBAL_DATA['covmat'][nplet][:,nplet]


def _f(n:int, r:int):
    n_fact = np.math.factorial(n)
    r_fact = np.math.factorial(r)
    n_r_fact = np.math.factorial(n - r)
    return int(n_fact / (r_fact * n_r_fact))


def _chunk_generator(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def _n_system_partitions(elids, m, n):
    """ Computes all the partitions of a N-element system, where 'elids' are the indexes of the system elements.
    
    INPUT
    elids: Vector with N entries, where each entry is the element id.
    if op=1, removes 1-element subsets, if =0 does nothing.
    
    OUTPUTS
    linparts: list of partitions, where each list entry is a vector with the elements of the partition. Sorted by ascending sizes of the partitions.
    psizes: array with the respective sizes of the partitions on linparts. Sorted as linparts.
    """
    N = len(elids)

    assert n<=N, "Cant calculate partitions of size larger than the system size (n > len(elids))"

    for p in range(m, n+1):
        for part in combinations(elids, p):
            yield list(part)        


def get_size(part):
    return len(part)


def multi_order_meas(data, min_n=2, max_n=None, batch_size=1000000, only_synergestic=False, output_path='./'):
    """    
    data = T samples x N variables matrix    
    
    """

    sart_time = time.time()

    # make device cpu if not cuda available or cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    T, N = np.shape(data)
    n = N if max_n is None else max_n

    assert n<N, "max_n must be lower than len(elids). max_n >= len(elids))"
    assert min_n<=n, "min_n must be lower or equal than max_n. min_n > max_n"

    # Gaussian Copula of data
    _, thiscovmat = data2gaussian(data)

    TD_DTC_GLOBAL_DATA['covmat'] = thiscovmat

    data_ready_time = time.time()

    psize_times = []
    covmat_times = []

    # To compute using pytorch, we need to compute each order separately
    for order in range(min_n, n+1):

        linparts_generator = _n_system_partitions(range(N), order, order)
        chunked_linparts_generator = _chunk_generator(linparts_generator, batch_size)

        total_chunks = _f(N, order) // batch_size + 1
        pbar = tqdm(enumerate(chunked_linparts_generator), total=total_chunks)
        for i, chunk_linpart_generator in pbar:

            pbar.set_description(f'Processing chunk {i}: computing partitions')
            linparts = [part for part in chunk_linpart_generator]

            if total_chunks > 1 and i < (total_chunks - 1):
                time_start_psizes = time.time()
            
            pbar.set_description(f'Processing chunk {i}: computing nplets sizes')
            with Pool(cpu_count()) as p:
                psizes = np.array(p.map(get_size, linparts))

            if total_chunks > 1 and i < (total_chunks - 1):
                time_betwen_psize_covmat = time.time()
                psize_times.append(time_betwen_psize_covmat - time_start_psizes)

            pbar.set_description(f'Processing chunk {i}: computing sub-covmats')
            with Pool(cpu_count()) as p:
                all_covmats = np.array(p.map(_get_covmat, linparts))
            
            if total_chunks > 1 and i < (total_chunks - 1):
                time_end = time.time()
                covmat_times.append(time_end - time_betwen_psize_covmat)

            pbar.set_description(f'Processing chunk {i}: computing nplets')
            nplets_tc, nplets_dtc, nplets_o, nplets_s = _get_tc_dtc_from_batched_covmat(all_covmats, order, T, device)

            pbar.set_description(f'Saving chunk {i}')
            _save_to_csv(
                os.path.join(output_path, f'nplets_{order}_{i}.csv'),
                nplets_tc, nplets_dtc,
                nplets_o, nplets_s,
                linparts, psizes,
                N, only_synergestic=only_synergestic
            )

    TD_DTC_GLOBAL_DATA['covmat'] = None

    # print total elapsed time
    print('Total time: ', (time.time() - sart_time) / 60, ' minutes')
    print('Data ready time: ', (data_ready_time - sart_time) / 60, ' minutes')
    print('psize times mean: ', np.mean(psize_times) / 60, ' minutes')
    print('covmat times mean: ', np.mean(covmat_times) / 60, ' minutes')

    
def nd_xcorr(data,maxlags):
    """    
    data = T samples x N variables matrix. T>N    
    
    """    
    T,N = np.shape(data)
    lags = signal.correlation_lags(maxlags, maxlags)
    sel_lags = range(T-maxlags,T+maxlags-1)
    npairs = int(N*(N-1)/2)
    pair_list = np.zeros((npairs,2))
    xcorr = np.zeros((npairs,len(sel_lags)))
    cont=0
    for i in range(0,N):
        for j in range(i+1,N):
            pair_list[cont,:] = [i,j]
            x = data[:,i]
            y = data[:,j]
            # x = data[:,i] - np.mean(data[:,i])
            # y = data[:,j] - np.mean(data[:,j])
            # x = data[:,i]/np.linalg.norm(data[:,i])
            # y = data[:,j]/np.linalg.norm(data[:,j])
            # x = stats.zscore(data[:,i])
            # y = stats.zscore(data[:,j])
            # x = x/np.linalg.norm(x)
            # y = y/np.linalg.norm(y)            
            corr = signal.correlate(x, y)            
            corr = corr/np.sqrt(np.sum(np.abs(x)**2)*np.sum(np.abs(y)**2))
    
            xcorr[cont,:] = corr[sel_lags]
    
            cont=cont+1
    
    return lags,xcorr,pair_list


def hilphase(y1,y2):
    sig1_hill=sig.hilbert(y1)
    sig2_hill=sig.hilbert(y2)
    pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
               np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
    phase = np.angle(pdt)

    return phase


def parts_idxs(parts):
    parts_list = list()
    for part in parts:
        parts_list.append(part)
    return parts_list


def random_k_ids(N,k,nints,prob='uni'):
    """Generates nints random combinations of k elements in a N-sized system    
    """    
    if k<5: # brute force for 4 or less subnetworks
        if math.comb(N,k)<=nints:
            rand_ints=list(combinations(range(N), k))
            return rand_ints
    
    # Generating probabilites for random sampling of sub-networks ids
    # See https://ethankoch.medium.com/incredibly-fast-random-sampling-in-python-baf154bd836a
    
    if isinstance(prob,str): # prob is string            
        if prob=='uni':
            probabilities = np.ones(N)/N # uniform probs of each id 
            
        elif prob=='rand':
            probabilities = np.random.random(N)
            probabilities /= np.sum(probabilities)
            
    else:
        probabilities = prob # uses user input of probs
    
    replicated_probabilities = np.tile(probabilities, (nints, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    rand_ints = np.argpartition(shifted_probabilities, k, axis=1)[:, :k]
    
    return rand_ints