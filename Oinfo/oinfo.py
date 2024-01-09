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


TD_DTC_GLOBAL_DATA = {
    'covmat': None,
    'linparts': None,
    'bc1': None,
    'T': None
}


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
    return 0.5*np.log((2*np.pi*np.exp(1))**(x)*y)


def _all_min_1_ids(N):
    return [np.setdiff1d(range(N),x) for x in range(N)]


def _get_tc_dtc_from_covmat(i: int):

    # Obtaining global data
    T = TD_DTC_GLOBAL_DATA['T']
    bc1 = TD_DTC_GLOBAL_DATA['bc1']
    full_covmat = TD_DTC_GLOBAL_DATA['covmat']
    linparts = TD_DTC_GLOBAL_DATA['linparts']

    # Compute parameters from shared memory data
    nplet = linparts[i]
    covmat = full_covmat[nplet][:,nplet]
    N = len(nplet)
    allmin1 = _all_min_1_ids(N)
    bcN = _gauss_ent_biascorr(N,T)
    bcNmin1 = _gauss_ent_biascorr(N-1,T)

    # allmin1 = all_min_1_ids(N)
    detmv = np.linalg.det(covmat) # determinant
    detmv_min_1 = np.array([np.linalg.det(covmat[ids][:,ids]) for ids in allmin1]) # determinant of N-1 subsystems
    single_vars = np.diag(covmat)

    var_ents = _gauss_ent_est(1.0,single_vars) - bc1
    sys_ent = _gauss_ent_est(N,detmv) - bcN
    ent_min_one = _gauss_ent_est(N-1.0,detmv_min_1) - bcNmin1

    tc = np.sum(var_ents) - sys_ent
    dtc = np.sum(ent_min_one) - (N-1.0)*sys_ent
    
    return tc, dtc, i


def _get_nplets_mvinfo(linparts, covmat, T, n_jobs=None):
    nnplets = len(linparts)
    bc1 = _gauss_ent_biascorr(1,T)
    tc = np.ndarray(nnplets)
    dtc = np.ndarray(nnplets)

    # Check global data is not being used
    if any([v is not None for v in TD_DTC_GLOBAL_DATA.values()]):
        raise Exception('Global data is being used. Please check that no other process is using it.')
    
    # Create shared memory data
    TD_DTC_GLOBAL_DATA['covmat'] = covmat
    TD_DTC_GLOBAL_DATA['linparts'] = linparts
    TD_DTC_GLOBAL_DATA['bc1'] = bc1
    TD_DTC_GLOBAL_DATA['T'] = T

    if n_jobs is not None:
        
        # Parallelization
        p = Pool(n_jobs if n_jobs > 0 else cpu_count())

        for (cur_tc, cur_dtc, i) in tqdm(p.imap_unordered(_get_tc_dtc_from_covmat, range(nnplets), chunksize=10000), total=nnplets, leave=False, desc='Computing nplets'):
            tc[i], dtc[i] = cur_tc, cur_dtc
        p.close()
    else:

        # No parallelization
        for i in tqdm(range(nnplets), total=nnplets, leave=False):
            tc[i], dtc[i], _ = _get_tc_dtc_from_covmat(i)

    # remove shared memory data
    TD_DTC_GLOBAL_DATA['covmat'] = None
    TD_DTC_GLOBAL_DATA['linparts'] = None
    TD_DTC_GLOBAL_DATA['bc1'] = None
    TD_DTC_GLOBAL_DATA['T'] = None


    return tc,dtc


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


def multi_order_meas(data, min_n=2, max_n=None, n_jobs=None, chunksize=25000000, only_synergestic=False, output_path='./'):
    """    
    data = T samples x N variables matrix    
    
    """
    
    T, N = np.shape(data)
    n = N if max_n is None else max_n

    assert n<N, "max_n must be lower than len(elids). max_n >= len(elids))"
    assert min_n<=n, "min_n must be lower or equal than max_n. min_n > max_n"

    # Gaussian Copula of data
    _, thiscovmat = data2gaussian(data)

    linparts_generator = _n_system_partitions(range(N), min_n, n)
    chanked_linparts_generator = _chunk_generator(linparts_generator, chunksize)

    # iterate overt linparts_generator by chunks of chunksize
    total_parts = int(np.sum([_f(N, i) for i in range(min_n, n+1)]))
    pbar = tqdm(enumerate(chanked_linparts_generator), total=total_parts//chunksize)
    for i, chunk_linpart_generator in pbar:

        pbar.set_description(f'Processing chunk {i}: computing partitions')
        linparts = [part for part in chunk_linpart_generator]

        pbar.set_description(f'Processing chunk {i}: computing nplets sizes')
        psizes = np.array([len(part) for part in linparts])

        pbar.set_description(f'Processing chunk {i}: computing nplets')
        nplet_tc, nplet_dtc = _get_nplets_mvinfo(linparts, thiscovmat, T, n_jobs=n_jobs)
        nplet_o = nplet_tc - nplet_dtc
        nplet_s = nplet_tc + nplet_dtc

        print(f'Saving chunk {i}')
        _save_to_csv(
            os.path.join(output_path, f'nplets_{i}.csv'),
            nplet_tc, nplet_dtc,
            nplet_o, nplet_s,
            linparts, psizes,
            N, only_synergestic=only_synergestic
        )

    
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