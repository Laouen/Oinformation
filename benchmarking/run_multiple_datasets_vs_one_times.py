import pandas as pd
import numpy as np
import time
from tqdm import trange
import torch

import argparse

from thoi.measures.gaussian_copula import multi_order_measures
from thoi.measures.gaussian_copula_hot_encoded import multi_order_measures_hot_encoded
from thoi.commons import gaussian_copula_covmat

def main(min_D, step_D, max_D, min_T, step_T, max_T, min_N, step_N, max_N, min_bs, step_bs, max_bs, min_order, max_order, indexing_method, device, output_path):

    """
        T = number of samples
        N = number of features
    """
    
    multi_order_measures_func = multi_order_measures if indexing_method == 'indexes' else multi_order_measures_hot_encoded

    max_T = min_T if max_T is None else max_T
    max_N = min_N if max_N is None else max_N
    max_bs = min_bs if max_bs is None else max_bs
    min_order = 3 if min_order is None else min_order

    def delete_batch(*args):
        for arg in args:
            del arg
        torch.cuda.empty_cache()

    def empty_cache(arg):
        torch.cuda.empty_cache()

    assert device == 'cpu' or torch.cuda.is_available(), 'GPU is not available'
    device = torch.device(device)

    rows = []
    for T in trange(min_T, max_T+1, step_T, leave=False, desc='T'): 
        for N in trange(min_N, max_N+1, step_N, leave=False, desc='N'):
            for batch_size in trange(min_bs, max_bs+1, step_bs, leave=False, desc='batch_size'):
                for D in trange(min_D, max_D+1, step_D, leave=False, desc='D'):
                    max_order = N if max_order is None else max_order
                    assert min_order <= max_order, f'min_order must be <= max_order. {min_order} > {max_order}'

                    X = np.random.rand(T, N)
                    X_list = [X.copy() for _ in range(D)]

                    start = time.time()
                    multi_order_measures_func(
                        X_list, min_order=min_order, max_order=max_order,
                        batch_size=batch_size, device=device,
                        batch_data_collector=delete_batch,
                        batch_aggregation=empty_cache
                    )
                    delta_t = time.time() - start
                    rows.append(['timeseries', D, T, N, batch_size, min_order, max_order, str(device), delta_t])

                    covmat = gaussian_copula_covmat(X)
                    covmat_list = [covmat.copy() for _ in range(10)]

                    start = time.time()
                    multi_order_measures_func(
                        covmat_list, min_order=min_order, max_order=max_order,
                        batch_size=batch_size, device=device,
                        batch_data_collector=delete_batch,
                        batch_aggregation=empty_cache
                    )
                    delta_t = time.time() - start
                    rows.append(['covmat', D, T, N, batch_size, min_order, max_order, str(device), delta_t])

                    pd.DataFrame(
                        rows,
                        columns=['input_type', 'D', 'T', 'N', 'batch_size', 'min_order', 'max_order', 'device', 'time']
                    ).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Test run time for HOI O information')
    parser.add_argument('--min_D', type=int, help='Number of datasets to pass as input', default=1)
    parser.add_argument('--step_D', type=int, help='Step for the number of datasets to pass as input', default=1)
    parser.add_argument('--max_D', type=int, help='Max number of datasets to pass as input', default=10)
    parser.add_argument('--min_T', type=int, help='Min number of samples')
    parser.add_argument('--step_T', type=int, help='Step for number of samples', default=1)
    parser.add_argument('--max_T', type=int, help='Max number of samples', default=None)
    parser.add_argument('--min_N', type=int, help='Min number of features')
    parser.add_argument('--step_N', type=int, help='Step for number of features', default=1)
    parser.add_argument('--max_N', type=int, help='Max number of features', default=None)
    parser.add_argument('--min_bs', type=int, help='Min batch size')
    parser.add_argument('--step_bs', type=int, help='Step for batch size', default=1)
    parser.add_argument('--max_bs', type=int, help='Max batch size', default=None)
    parser.add_argument('--min_order', type=int, help='Min size of the n-plets')
    parser.add_argument('--max_order', type=int, help='Max size of the n-plets', default=None)
    parser.add_argument('--indexing_method', default='indexes', help='Indexing method to use. hot_encoded or indexes')
    parser.add_argument('--device', type=str, default='cpu', help='The device to use. cpu or cuda. Note: cuda is only available if torch.cuda.is_available()')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')

    args = parser.parse_args()

    main(
        args.min_D, args.step_D, args.max_D,
        args.min_T, args.step_T, args.max_T,
        args.min_N, args.step_N, args.max_N,
        args.min_bs, args.step_bs, args.max_bs,
        args.min_order, args.max_order,
        args.indexing_method, args.device, args.output_path
    )