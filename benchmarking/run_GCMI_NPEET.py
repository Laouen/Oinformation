from itertools import combinations
import pandas as pd
import numpy as np
import math
import time
from tqdm import tqdm

import argparse

from npeet import entropy_estimators as ee

from Oinfo import o_information

import os
import sys

GCMI_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'libraries')  # This gets the directory where main.py is located
print('GCMI dir', GCMI_dir)
sys.path.append(GCMI_dir)

from gcmi.python import gcmi

class systemPartsDataset:
    def __init__(self, X, order):
        """
        Initialize the dataset for generating combinations of system parts.

        Args:
            X (np.ndarray): A 2D array of shape (T samples, N features) representing the dataset.
            order (int): The number of features to include in each combination.
        """
        
        self.X = X
        self.order = order
        self.N = self.X.shape[1]
        self.linparts_generator = combinations(range(self.N), self.order)

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return math.comb(self.N, self.order)

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            X_part (np.ndarray): The subset of the data matrix corresponding to the current combination, shape (T, order).
        """
        for part in self.linparts_generator:
            yield part, self.X[:,part]


def order_o_information(X, order, entropy_func):
    """
    Compute O-information for all n-plets of a given order.

    Args:
        X (np.ndarray): The input X matrix (T samples x N variables).
        entropy_func (callable): The entropy function to be used for calculation.
        order (int): The order of combinations to consider.

    Raises:
        AssertionError: If order is greater than N.
    """
    
    N = np.shape(X)[1]
    
    assert order <= N, f"ValueError: order must be lower or equal than N. {order} >= {N}"

    single_exclusions_mask = (np.ones((order, order)) - np.eye(order)).astype(bool)
    all_individual_entropies = np.array([entropy_func(X[:, i]) for i in range(N)])
    dataset = systemPartsDataset(X, order)

    for (idxs, X) in tqdm(dataset, total=len(dataset), leave=False, desc='n-plet'):
        o_information(
            X,
            entropy_func,
            single_exclusions_mask=single_exclusions_mask,
            individual_entropies=all_individual_entropies[idxs] # avoid recomputing individual entropies
        )


def main(min_T, step_T, max_T, min_N, step_N, max_N, min_order, max_order, estimator, output_path):

    """
        T = number of samples
        N = number of features
    """

    max_T = min_T if max_T is None else max_T
    max_N = min_N if max_N is None else max_N
    max_order = min_order if max_order is None else max_order

    assert min_order <= max_order, f'min_order must be <= max_order. {min_order} > {max_order}'

    if estimator == 'npeet':
        o_estimator = ee.entropy

    elif estimator == 'gcmi':
        o_estimator = lambda X: gcmi.ent_g(X.T)

    rows = []
    for T in tqdm(range(min_T, max_T+1, step_T), leave=False, desc='T'): 
        for N in tqdm(range(min_N, max_N+1, step_N), leave=False, desc='N'):

            X = np.random.rand(T, N)

            for order in tqdm(range(min_order, max_order+1), leave=False, desc='Order'):

                start = time.time()
                order_o_information(X, order, o_estimator)
                delta_t = time.time() - start

                rows.append([estimator, T, N, order, delta_t])

                pd.DataFrame(
                    rows,
                    columns=['estimator', 'T', 'N', 'order', 'time']
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
    parser.add_argument('--estimator', type=str, choices=['gcmi', 'npeet'])
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')

    args = parser.parse_args()

    main(
        args.min_T, args.step_T, args.max_T,
        args.min_N, args.step_N, args.max_N,
        args.min_order, args.max_order,
        args.estimator,
        args.output_path
    )