import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from tqdm import tqdm

# This script generates multivariate normal distributed data for multiple given sample sizes and dimensionality
# The data is saved in a csv file with the following naming convention: nsamples-<sample_size>_nvars-<n_vars>.csv

def generate_data(sample_sizes, n_vars, output_path):

    # generate random covariance matrix
    cov = np.random.rand(n_vars, n_vars)
    cov = np.dot(cov, cov.T)

    mean = np.zeros(n_vars)

    for sample_size in tqdm(sample_sizes, desc='Generating data', leave=False):
        data = np.random.multivariate_normal(mean, cov, sample_size)
        pd.DataFrame(data).to_csv(
            os.path.join(output_path, f'nsamples-{sample_size}_nvars-{n_vars}.csv'),
            sep=',',
            index=False,
            header=False
        )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sample_sizes', type=int, nargs='+', help='List of sample sizes')
    parser.add_argument('--n_vars', type=int, help='Number of variables')
    parser.add_argument('--output_path', type=str, help='Output path to save the data')
    args = parser.parse_args()

    generate_data(args.sample_sizes, args.n_vars, args.output_path)
