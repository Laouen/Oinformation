from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from tqdm import tqdm

from thoi.heuristics.simulated_annealing_multi_order import simulated_annealing_multi_order
from thoi.measures.gaussian_copula import nplets_measures

def main(path_covariance_matrix: str, output_path: str):
    
    Path(output_path).mkdir(parents=True, exist_ok=True)

    covmat = np.load(path_covariance_matrix)
    N = covmat.shape[0]
    
    T = 100000
    X = np.random.multivariate_normal(np.zeros(N), covmat, T)
    
    n = N // 5
    npletas_subsystems = np.stack([np.arange(i, i+n) for i in range(0, N, n)])
    measures_subsystems = nplets_measures(X, nplets=npletas_subsystems)
    
    np.save(f'{output_path}/measures_sub_systems.npy', measures_subsystems)

    for step_size in tqdm([1, 2, 3, 4, 5], desc='step_size'):
        
        best_solution, best_energy = simulated_annealing_multi_order(
            X, repeat=1000, max_iterations=1000,
            step_size=step_size,
            use_cpu=True
        )
        
        # save best solution to a numpy file
        np.save(f'{output_path}/simulated_annealing_multi_order_stepsize-{step_size}_best_solution.npy', best_solution)
        np.save(f'{output_path}/simulated_annealing_multi_order_stepsize-{step_size}_best_oinfo.npy', best_energy)


if __name__ == '__main__':
    parser = ArgumentParser('run simulated annealing multi order')
    parser.add_argument('--path_covariance_matrix', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args.path_covariance_matrix, args.output_path)