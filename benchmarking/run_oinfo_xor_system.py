import pandas as pd
from tqdm import tqdm
import argparse

import formulas
import systems

def main(output_path: str, T: int, n_repeat: int):

    nplets = [
        ['X1','X2','Zxor'],
    ]

    value_range = [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]

    dfs = []
    for alpha in tqdm(value_range, leave=False, desc='alpha'):           
        rows = []
        for _ in tqdm(range(n_repeat), leave=False, desc='repet'):
            data = systems.generate_continuos_xor(alpha=alpha, T=T)

            for nplet in tqdm(nplets, leave=False, desc='nplet'):
                name = '-'.join(nplet)

                # (n_samples, n_variables)
                X = data[nplet].values

                rows.append({
                    'n-plet': name,
                    'method': 'THOI',
                    'O-information': formulas.o_information_thoi(X)
                })

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

        df = pd.DataFrame(rows)
        df = df.groupby(['n-plet', 'method']).mean().reset_index()
        df['alpha'] = alpha

        dfs.append(df)
        pd.concat(dfs).to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate flat systems and calculate the o information on the systems')
    parser.add_argument('--output_path', type=str, help='Path of the .tsv file where to store the results')
    parser.add_argument('--T', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--n_repeat', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.T, args.n_repeat)