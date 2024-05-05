import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
from pathlib import Path

def generate_flat_system(alpha=1.0, beta=1.0, gamma=1.0, T=1000):
    # Generate random samples
    Z00, Z01, Z1, Z2, Z3, Z4, Z5, Z6 = [np.random.normal(0, 1, T) for _ in range(8)]

    # Define the variables
    X1 = alpha*np.log(np.abs(Z00) + 1) + beta*Z01 + gamma*Z1
    X2 = alpha*Z00                     + beta*Z01 + gamma*Z2
    X3 = alpha*np.power(Z00,2)         + beta*Z01 + gamma*Z3
    X4 = alpha*np.exp(Z00)             + beta*Z01 + gamma*Z4
    X5 = alpha*np.sin(Z00)             + beta*Z01 + gamma*Z5
    X6 = alpha*np.cos(Z00)             + beta*Z01 + gamma*Z6

    return pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6,
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'Z4': Z4, 'Z5': Z5, 'Z6': Z6,
        'Z00': Z00, 'Z01': Z01
    })

def generate_herarchical_system(alpha=1.0, beta=1.0, T=1000):
    # Generate base random variables
    Z1, Z2, Z3, Z4, Z5, Z6 = np.random.normal(0, 1, (6, T))

    # Generate dependent variables
    X1 = Z1
    X2 = Z2 + alpha*np.exp(Z3)
    X3 = (np.log(alpha*np.abs(Z2) + 1) + 1) * Z3
    X4 = np.sin(Z4) + alpha*np.cos(Z5)
    X5 = (Z4 / (Z4-alpha*(Z4+1))) * np.sin(Z5)
    X6 = Z6 + beta*(np.sin(X2 + X3) + np.cos(X4 + X5))

    return pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6,
        'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'Z4': Z4, 'Z5': Z5, 'Z6': Z6
    })

SYSTEMS = {
    'flat': generate_flat_system,
    'hierarchical': generate_herarchical_system
}

def main(output_path: str, T: int):

    Path(output_path).mkdir(parents=True, exist_ok=True)

    for system in tqdm(['flat', 'hierarchical'], leave=False, desc='system'):
        for alpha in tqdm(np.arange(0.1, 1.05, 0.1), leave=False, desc='alpha'):
            for beta in tqdm(np.arange(0.1, 1.05, 0.1), leave=False, desc='beta'):
                
                if system == 'hierarchical':
                    df = generate_herarchical_system(alpha, beta, T)
                    df.to_csv(os.path.join(output_path, f'system-{system}_alpha-{alpha}_beta-{beta}_t-{T}.tsv'), sep='\t', index=False)
                
                else:
                    
                    for gamma in tqdm(np.arange(0.1, 1.05, 0.1), leave=False, desc='gamma'):
                        df = generate_flat_system(alpha, beta, gamma, T)
                        df.to_csv(os.path.join(output_path, f'system-{system}_alpha-{alpha}_beta-{beta}_gamma-{gamma}_t-{T}.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate flat or hierarchical systems and save them into tsv files')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--T', type=int, default=10000, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.T)
                        

                    