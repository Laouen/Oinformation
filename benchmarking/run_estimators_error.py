import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import systems

# TODO: change
def generate_covariance_matrix(dim, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)  # Set random seed for reproducibility
    
    # Generate a random matrix A
    A = np.random.randn(dim, dim)
    
    # Calculate the covariance matrix as A * A'
    covariance_matrix = np.dot(A, A.T)
    
    return covariance_matrix

def gaussian_entropy(covariance):
    n_variables = covariance.shape[0]
    cov_det = np.linalg.det(covariance) 
    return 0.5 * np.log((2 * np.pi * np.e)**n_variables * cov_det)

def main(output_path: str, n_repeat: int):

    rows = []
    for n_variables in tqdm(range(3,10), leave=False, desc='n_variables'):
        for _ in tqdm(range(n_repeat), leave=False, desc='repeat'):

            mean = np.random.randint(n_variables)
            covariance = generate_covariance_matrix(n_variables)
            real_entropy = gaussian_entropy(covariance)
            
            for T in tqdm([10**i for i in range(1,10)], leave=False, desc='n_samples'):
                X = np.random.multivariate_normal(mean, covariance, T)
                rows.append([
                    n_variables, T, real_entropy,
                    systems.npeet_entropy(X),
                    systems.gcmi_entropy(X)
                ])
    
    pd.DataFrame(
        rows,
        columns=['n_variables','n_samples','real_entropy','gcmi_entropy','npeet_entropy']
    ).to_csv(output_path, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate hierarchical systems and calculate the o information on the systems')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--n_repeat', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.n_repeat)