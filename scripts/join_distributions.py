import pandas as pd
from argparse import ArgumentParser
from glob import glob
import os

def split_db(file_paths):

    for distribution in ["beta","exp","uniform"]:
        for alpha in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]:
            data_hh = pd.read_csv(os.path.join(file_paths, f'hh_{distribution}_alpha-{alpha}.csv'), sep=',', header=None)
            data_tt = pd.read_csv(os.path.join(file_paths, f'tt_{distribution}_alpha-{alpha}.csv'), sep=',', header=None)

            print('data_hh.shape', data_hh.values.shape)
            print('data_tt.shape', data_tt.values.shape)

            # Concatenate the data column wise
            data = pd.concat([data_hh, data_tt], axis=1)

            print('data.shape', data.values.shape)

            # Save the data as a .csv file wihtout the column index and header
            csv_file = os.path.join(file_paths, f'joint_{distribution}_alpha-{alpha}.csv')
            data.to_csv(csv_file, index=False, header=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file_paths', type=str, required=True)
    args = parser.parse_args()

    split_db(args.file_paths)