from tqdm import tqdm
import argparse
import numpy as np
import os
from pathlib import Path

from systems import generate_relu_sistem

def main(output_path: str, pow_factor: float, T: int, n_repeat: int):
    
    Path(output_path).mkdir(parents=True, exist_ok=True)

    value_range = [
        0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99
    ]

    for alpha in tqdm(value_range, leave=False, desc='alpha'):
        beta = 1 - alpha
        #for beta in tqdm(value_range, leave=False, desc='beta'):        
        for i in tqdm(range(n_repeat), leave=False, desc='repet'):
            data = generate_relu_sistem(alpha=alpha, beta=beta, pow_factor=pow_factor, T=T)
            # save data in numpy format
            np.save(os.path.join(output_path, f'system-smoothSoftRelu_alpha-{alpha:.2f}_beta-{beta:.2f}_pow-{pow_factor:.1f}_T-{T}_repeat-{i}.npy'), data.values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate flat systems and calculate the o information on the systems')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--pow_factor', type=float, default=0.5, help='The factor to power the ReLU X1 and X2. default 0.5 for square root')
    parser.add_argument('--T', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--n_repeat', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.pow_factor, args.T, args.n_repeat)