import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import systems
from sklearn.datasets import make_spd_matrix


def generate_covariance_matrix(n_variables):

    mean = np.random.uniform(0, 10, n_variables)
    covariance = make_spd_matrix(n_variables)

    return mean, covariance

    
def gaussian_entropy(covariance):
    n_variables = covariance.shape[0]
    cov_det = np.linalg.det(covariance) 
    return 0.5 * np.log((2 * np.pi * np.e)**n_variables * cov_det)

def main(output_path: str, n_repeat: int):

    rows = []
    for n_variables in tqdm(range(3,11), leave=False, desc='n_variables'):
        for _ in tqdm(range(n_repeat), leave=False, desc='repeat'):

            mean, covariance = generate_covariance_matrix(n_variables)
            real_entropy = gaussian_entropy(covariance)
            
            for T in tqdm([10**i + 10**(k*i)//2 for i in range(1,7) for k in range(2)], leave=False, desc='n_samples'):
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