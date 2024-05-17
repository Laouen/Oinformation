import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
import time
import argparse

HOI_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'libraries/HOI_toolbox')  # This gets the directory where main.py is located
print('HOI Toolbox dir', HOI_dir)
sys.path.append(HOI_dir)

from toolbox.Oinfo import exhaustive_loop_zerolag

ESTIMATOR_FUNC = {
    'gcmi': 'GC',
    'lin_est': 'LIN-EST'
}

def main(min_T, step_T, max_T, min_N, step_N, max_N, min_order, max_order, estimator, output_path):

    """
        T = number of samples
        N = number of features
    """

    max_T = min_T if max_T is None else max_T
    max_N = min_N if max_N is None else max_N
    max_order = min_order if max_order is None else max_order

    assert min_order <= max_order, f'min_order must be <= max_order. {min_order} > {max_order}'

    config = {
        "higher_order": True,
        "estimator": estimator,
        "n_best": 10, 
        "nboot": 10
    }

    rows = []
    for T in tqdm(range(min_T, max_T+1, step_T), leave=False, desc='T'): 
        for N in tqdm(range(min_N, max_N+1, step_N), leave=False, desc='T'):

            X = np.random.rand(N, T)

            for order in tqdm(range(min_order, max_order+1), leave=False, desc='Order'):

                config['minsize'] = order
                config['maxsize'] = order

                start = time.time()
                exhaustive_loop_zerolag(X, config)
                delta_t = time.time() - start

                rows.append(['HOI', ESTIMATOR_FUNC[estimator], T, N, order, delta_t])

                # Save to disk current data to avoid data lost if script stops
                pd.DataFrame(
                    rows,
                    columns=['library', 'estimator' ,'T','N','order', 'time']
                ).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Test run time for HOI O information')
    parser.add_argument('--min_T', type=int, help='Min number of samples')
    parser.add_argument('--step_T', type=int, help='Step for number of samples', default=1)
    parser.add_argument('--max_T', type=int, help='Max number of samples', default=None)
    parser.add_argument('--min_N', type=int, help='Min number of features')
    parser.add_argument('--step_N', type=int, help='Step for number of features', default=1)
    parser.add_argument('--max_N', type=int, help='Max number of features', default=None)
    parser.add_argument('--min_order', type=int, help='Min size of the n-plets')
    parser.add_argument('--max_order', type=int, help='Max size of the n-plets', default=None)
    parser.add_argument('--estimator', type=str, choices=['gcmi', 'lin_est'])
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')

    args = parser.parse_args()

    main(
        args.min_T, args.step_T, args.max_T,
        args.min_N, args.step_N, args.max_N,
        args.min_order, args.max_order,
        args.estimator,
        args.output_path
    )