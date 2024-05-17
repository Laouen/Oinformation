import pandas as pd
from tqdm import tqdm
import argparse

from Oinfo import o_information
import systems

def main(output_path: str, T: int, n_repeat: int):

    nplets = [
        # without synergistic and redundant source
        ['X1','X2','X3'],
        ['X1','X2','X3','X4'],
        ['X1','X2','X3','X4', 'X5'],
        ['X1','X2','X3','X4', 'X5', 'X6'],

        # with synergistic and redundant source
        #['Z00','Z01','X1','X2','X3'],
        #['Z00','Z01','X1','X2','X3','X4'],
        #['Z00','Z01','X1','X2','X3','X4', 'X5'],
        ['Z00','X1','X2','X3','X4', 'X5', 'X6'],
        ['Z01','X1','X2','X3','X4', 'X5', 'X6'],
        ['Z00','Z01','X1','X2','X3','X4', 'X5', 'X6']
    ]

    value_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    dfs = []
    for gamma in tqdm([0.1], leave=False, desc='gamma'):
        for alpha in tqdm(value_range, leave=False, desc='alpha'):
            for beta in tqdm(value_range, leave=False, desc='beta'):            
                rows = []
                for _ in tqdm(range(n_repeat), leave=False, desc='repet'):
                    data = systems.generate_flat_system(alpha=alpha, beta=beta, gamma=gamma, T=T)

                    for nplet in tqdm(nplets, leave=False, desc='nplet'):
                        name = '-'.join(nplet)
                        X = data[nplet].values

                        rows.append({
                            'n-plet': name,
                            'method': 'THOI',
                            'O-information': systems.o_information_thoi(X)
                        })

                        rows.append({
                            'n-plet': name,
                            'method': 'NPEET',
                            'O-information': o_information(X, systems.npeet_entropy)
                        })

                        rows.append({
                            'n-plet': name,
                            'method': 'GCMI',
                            'O-information': o_information(X, systems.gcmi_entropy)
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
    parser.add_argument('--n_repeat', type=int, default=20, help='Number of samples to generate')

    args = parser.parse_args()

    main(args.output_path, args.T, args.n_repeat)