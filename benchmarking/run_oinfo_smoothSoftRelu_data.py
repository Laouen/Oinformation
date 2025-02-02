import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np
import os

from formulas import o_information_thoi
from systems import generate_relu_sistem

def main(data_path: str, output_path: str, pow_factor: float, T: int, n_repeat: int):

    nplets = [
        ['X1','X2','Z_syn','Z_red'],
        ['X1','X2','Z_syn'],
        ['X1','X2','Z_red'],
        ['X1','Z_red','Z_syn'],
        ['X2','Z_red','Z_syn']
    ]

    value_range = [
        0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99
    ]

    dfs = []
    for alpha in tqdm(value_range, leave=False, desc='alpha'):
        beta = 1 - alpha
        #for beta in tqdm(value_range, leave=False, desc='beta'):        
        rows = []
        for i in tqdm(range(n_repeat), leave=False, desc='repet'):
            # read data from numpy file
            data = np.load(os.path.join(data_path, f'system-smoothSoftRelu_alpha-{alpha:.2f}_beta-{beta:.2f}_pow-{pow_factor:.1f}_T-{T}_repeat-{i}.npy'))
            data = pd.DataFrame(data, columns=['X1', 'X2', 'Z_syn', 'Z_red'])
            
            for nplet in tqdm(nplets, leave=False, desc='nplet'):
                name = '-'.join(nplet)

                # (n_samples, n_variables)
                X = data[nplet].values

                rows.append({
                    'n-plet': name,
                    'method': 'THOI',
                    'O-information': o_information_thoi(X)
                })

                '''
                rows.append({
                    'n-plet': name,
                    'method': 'NPEET',
                    'O-information': formulas.o_information(X, formulas.npeet_entropy)
                })

                rows.append({
                    'n-plet': name,
                    'method': 'GCMI',
                    'O-information': formulas.o_information(X, formulas.gcmi_entropy)
                })
                '''

        df = pd.DataFrame(rows)
        df = df.groupby(['n-plet', 'method']).mean().reset_index()
        df['alpha'] = alpha
        df['beta'] = beta

        dfs.append(df)
        pd.concat(dfs).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate flat systems and calculate the o information on the systems')
    parser.add_argument('--data_path', type=str, help='Path of the .tsv files where to read the data')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--pow_factor', type=float, default=0.5, help='The factor to power the ReLU X1 and X2. default 0.5 for square root')
    parser.add_argument('--T', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--n_repeat', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.data_path, args.output_path, args.pow_factor, args.T, args.n_repeat)