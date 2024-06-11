import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import numpy as np

from sklearn.datasets import make_spd_matrix

import formulas


def is_pd(matrix):

    try:
        # Attempt to compute the Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True  # Successful decomposition means the matrix is positive definite
    except np.linalg.LinAlgError:
        return False  # If the decomposition fails, the matrix is not positive definite


def generate_covariance_matrix(n_variables):

    mean = np.random.uniform(0, 10, n_variables)
    covariance = make_spd_matrix(n_variables)

    if not is_pd(covariance):
        print(covariance)

    return mean, covariance


def generate_uniforme_bounds(n_variables):
    lower_bounds = np.random.randint(0, 10, size=n_variables)
    upper_bounds = np.random.randint(lower_bounds+1, lower_bounds+10)

    assert all([(u - l) >= 0 for l,u in zip(lower_bounds, upper_bounds)]), f'wrong lower and upper bounds. {lower_bounds}, {upper_bounds}'

    return lower_bounds, upper_bounds


ENTROPY_FUNC = {
    'gaussian': lambda params: formulas.gaussian_entropy(params[1]),
    'uniform': lambda params: formulas.uniform_entropy(*params),
    'dirichlet': formulas.dirichlet_entropy
}

OINFO_FUNC = {
    'gaussian': lambda params: formulas.o_information_gaussian(params[1]),
    'uniform': lambda params: formulas.o_information_uniform(*params),
    'dirichlet': formulas.o_information_dirichlet
}

DISTRIBUTION_PARAMS_FUNC = {
    'gaussian': generate_covariance_matrix,
    'uniform': generate_uniforme_bounds,
    'dirichlet': lambda n_variables: np.random.randint(1,10,n_variables)
}

RANDOM_SAMPLER = {
    'gaussian': lambda params,T: np.random.multivariate_normal(*params, T),
    'uniform': lambda params,T: np.random.uniform(*params, (T, len(params[0]))),
    'dirichlet': lambda a,T: np.random.dirichlet(a, T)
}

def main(output_path: str, distribution: str, n_repeat: int):

    rows = []
    for n_variables in tqdm(range(3,11), leave=False, desc='n_variables'):
        for _ in tqdm(range(n_repeat), leave=False, desc='repeat'):

            params = DISTRIBUTION_PARAMS_FUNC[distribution](n_variables)
            real_entropy = ENTROPY_FUNC[distribution](params)
            real_o_information = OINFO_FUNC[distribution](params)

            for T in tqdm([10**i + 10**(k*i)//2 for i in range(1,7) for k in range(2)], leave=False, desc='n_samples'):
                X = RANDOM_SAMPLER[distribution](params, T)

                # I use the try and catch because for some very particular casses gcmi can 
                # have a not positive defined covariance matrix and failes
                processed = False
                while(not processed):
                    try:
                        row = [
                            n_variables, T, real_entropy, real_o_information,
                            formulas.npeet_entropy(X),
                            formulas.gcmi_entropy(X),
                            formulas.o_information(X, formulas.npeet_entropy),
                            formulas.o_information(X, formulas.gcmi_entropy)
                        ]
                        processed = True
                    except np.linalg.LinAlgError as e:
                        tqdm.write(f'Error processing random sampler, try again: {e}')

                rows.append(row)
    
        pd.DataFrame(
            rows,
            columns=[
                'n_variables','n_samples','real_entropy','real_o_information',
                'npeet_entropy',
                'gcmi_entropy',
                'o_info_npeet', 
                'o_info_gcmi'
            ]
        ).to_csv(output_path, sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate hierarchical systems and calculate the o information on the systems')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--distribution', type=str, choices=['gaussian', 'uniform', 'dirichlet'], help='The distribution function to calculate estimator error')
    parser.add_argument('--n_repeat', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.distribution, args.n_repeat)