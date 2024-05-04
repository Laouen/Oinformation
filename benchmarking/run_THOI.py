import pandas as pd
import numpy as np
import time
from tqdm import tqdm

import argparse

from Oinfo import multi_order_meas_gc, multi_order_meas_knn

ESTIMATORS = {
    'gc': multi_order_meas_gc,
    'knn': multi_order_meas_knn
}

def main(min_T, max_T, min_N, max_N, min_order, max_order, estimator, batch_size, output_path):

    """
        T = number of samples
        N = number of features
    """

    max_T = min_T if max_T is None else max_T
    max_N = min_N if max_N is None else max_N
    max_order = min_order if max_order is None else max_order

    assert min_order <= max_order, f'min_order must be <= max_order. {min_order} > {max_order}'


    multi_order_meas = ESTIMATORS[estimator]


    rows = []
    for T in tqdm(range(min_T, max_T+1), leave=False, desc='T'): 
        for N in tqdm(range(min_N, max_N+1), leave=False, desc='N'):

            X = np.random.rand(T, N)

            for order in tqdm(range(min_order, max_order+1), leave=False, desc='Order'):

                start = time.time()
                multi_order_meas(X, order, order, batch_size)
                delta_t = time.time() - start

                rows.append(['ours_' + estimator, T, N, order, delta_t])

                pd.DataFrame(
                    rows,
                    columns=['estimator', 'T', 'N', 'order', 'time']
                ).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Test run time for HOI O information')
    parser.add_argument('--min_T', type=int, help='Min number of samples')
    parser.add_argument('--max_T', type=int, help='Max number of samples', default=None)
    parser.add_argument('--min_N', type=int, help='Min number of features')
    parser.add_argument('--max_N', type=int, help='Max number of features', default=None)
    parser.add_argument('--min_order', type=int, help='Min size of the n-plets')
    parser.add_argument('--max_order', type=int, help='Max size of the n-plets', default=None)
    parser.add_argument('--estimator', type=str, choices=['gc', 'knn'])
    parser.add_argument('--batch_size', type=int, default=1000000)
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')

    args = parser.parse_args()

    main(
        args.min_T, args.max_T,
        args.min_N, args.max_N,
        args.min_order, args.max_order,
        args.estimator, args.batch_size,
        args.output_path
    )