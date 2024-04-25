import numpy as np
import pandas as pd
from tqdm import tqdm
from .libraries.HOI_toolbox.toolbox.Oinfo import exhaustive_loop_zerolag

import time
import argparse

def main(min_T, max_T, min_N, max_N, min_order, max_order, estimator, output_path):

    """
        T = number of samples
        N = number of features
    """

    assert min_order <= max_order, f'min_order must be <= max_order. {min_order} > {max_order}'

    config = {
        "higher_order": True,
        "estimator": estimator,
        "n_best": 10, 
        "nboot": 0
    }

    rows = []
    for T in tqdm(range(min_T, max_T+1), leave=False, desc='T'): 
        for N in tqdm(range(min_N, max_N+1), leave=False, desc='T'):

            X = np.random.rand(N, T)

            for order in tqdm(range(min_order, max_order+1), leave=False, desc='Order'):

                config['minsize'] = order
                config['maxsize'] = order

                start = time.time()
                exhaustive_loop_zerolag(X, config)
                delta_t = time.time() - start

                rows.append([T, N, order, delta_t])

    pd.DataFrame(
        rows,
        columns=['HOI__' + estimator ,'T','N','order', 'time']
    ).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Test run time for HOI O information')
    parser.add_argument('--min_T', type=int, help='Min number of samples')
    parser.add_argument('--max_T', type=int, help='Max number of samples')
    parser.add_argument('--min_N', type=int, help='Min number of features')
    parser.add_argument('--max_N', type=int, help='Max number of features')
    parser.add_argument('--min_order', type=int, help='Min size of the n-plets')
    parser.add_argument('--max_order', type=int, help='Max size of the n-plets')
    parser.add_argument('--estimator', type='str', choices=['gcmi', 'lin_est'])
    parser.add_argument('-output_path', '--o', type=str, help='Path of the .tsv file where to store the results')

    args = parser.parse_args()

    main(
        args.min_T, args.max_T,
        args.min_N, args.max_N,
        args.min_order, args.max_order,
        args.estimator,
        args.output_path
    )