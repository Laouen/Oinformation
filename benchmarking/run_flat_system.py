import numpy as np
import pandas as pd

from npeet import entropy_estimators as ee

from Oinfo import o_information
from Oinfo.oinfo_gc import nplet_tc_dtc

from .run_generate_systems import generate_flat_system

from tqdm import tqdm

import argparse
import sys
import os

GCMI_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libraries')
sys.path.append(GCMI_dir)

from gcmi.python import gcmi


def main(output_path: str, T: int):

    nplets = [
        ['X1','X2','X3'],
        ['X1','X2','X3','X4'],
        ['X1','X2','X3','X4', 'X5'],
        ['X1','X2','X3','X4', 'X5', 'X6'],
    ]

    dfs = []
    for alpha in tqdm(np.arange(0.1, 1.05, 0.1), leave=False, desc='alpha'):
        for beta in tqdm(np.arange(0.1, 1.05, 0.1), leave=False, desc='beta'):
            for gamma in tqdm(np.arange(0.1, 1.05, 0.1), leave=False, desc='gamma'):
                rows = []
                for _ in tqdm(range(5), leave=False, desc='repet'):
                    data = generate_flat_system(alpha=alpha, beta=beta, gamma=gamma, T=T)

                    for nplet in tqdm(nplets, leave=False, desc='nplet'):
                        name = '-'.join(nplet)
                        X = data[nplet].values

                        rows.append({
                            'n-plet': name,
                            'method': 'THOI',
                            'O-information': nplet_tc_dtc(X)[2]
                        })

                        rows.append({
                            'n-plet': name,
                            'method': 'NPEET',
                            'O-information': o_information(X, ee.entropy)
                        })

                        rows.append({
                            'n-plet': name,
                            'method': 'GCMI',
                            'O-information': o_information(X, lambda X: gcmi.ent_g(X.T))
                        })

                df = pd.DataFrame(rows)
                df = df.groupby(['n-plet', 'method']).mean().reset_index()
                df['alpha'] = alpha
                df['beta'] = beta
                df['gamma'] = gamma

                dfs.append(df)
                pd.concat(dfs).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate flat systems and calculate the o information on the systems')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--T', type=int, default=10000, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.T)