import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
import os

def split_db(csv_file_pattern):

    for csv_file in glob(csv_file_pattern):

        print('csv_file', csv_file)

        # Load the .npy file
        data = pd.read_csv(csv_file, sep=',', header=None)

        # Print the shape of the data
        print('data.shape', data.values.shape)

        # Split data in the first 4 cols and the second 4 cols
        data1 = data.values[:, :4]
        data2 = data.values[:, 4:]

        print('data1.shape', data1.shape)
        print('data2.shape', data2.shape)
        
        # Save the data as a .csv file wihtout the column index and header
        df = pd.DataFrame(data1)
        csv_file1 = csv_file.replace('db', 'db1')
        df.to_csv(csv_file1, index=False, header=False)

        df = pd.DataFrame(data2)
        csv_file2 = csv_file.replace('db', 'db2')
        df.to_csv(csv_file2, index=False, header=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--csv_file_pattern', type=str, required=True)
    args = parser.parse_args()

    split_db(args.csv_file_pattern)