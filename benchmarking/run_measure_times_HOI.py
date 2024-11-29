import pandas as pd
import numpy as np
import time
from tqdm import trange
import argparse
import hoi

from memory_profiler import memory_usage


def main(min_T, step_T, max_T, min_N, step_N, max_N, min_order, max_order, method, output_path):

    """
        T = number of samples
        N = number of features
    """
    
    max_T = min_T if max_T is None else max_T
    max_N = min_N if max_N is None else max_N
    max_order = min_order if max_order is None else max_order

    assert min_order <= max_order, f'min_order must be <= max_order. {min_order} > {max_order}'

    rows = []
    for T in trange(min_T, max_T+1, step_T, leave=False, desc='T'): 
        for N in trange(min_N, max_N+1, step_N, leave=False, desc='N'):
            
            X = np.random.rand(T, N)

            for order in trange(min_order, max_order+1, leave=False, desc='Order'):

                if order > N:
                    continue

                try:
                    start = time.time()
                    h = hoi.metrics.Oinfo(X)
                    #mem_usage = memory_usage((h.fit, (order, order), {'method': method}), interval=0.1)
                    h.fit(order, order, method=method)
                    delta_t = time.time() - start
                    #max_mem = max(mem_usage)
                except MemoryError as me:
                    # Handle MemoryError
                    delta_t = -1
                    #max_mem = -1
                except Exception as e:
                    # Handle other exceptions
                    delta_t = -1
                    #max_mem = -1

                #rows.append(['HOI', method.upper(), T, N, order, delta_t, max_mem])
                rows.append(['HOI', method.upper(), T, N, order, delta_t])

                pd.DataFrame(
                    rows,
                    columns=['library', 'estimator', 'T', 'N', 'order', 'time']
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
    parser.add_argument('--method', default='gc', help='estimator to use for the entropy')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')

    args = parser.parse_args()

    main(
        args.min_T, args.step_T, args.max_T,
        args.min_N, args.step_N, args.max_N,
        args.min_order, args.max_order,
        args.method, args.output_path
    )