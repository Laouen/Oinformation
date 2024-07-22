import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
import os

def npy_to_csv(npy_file_dir):

    files = glob(os.path.join(npy_file_dir,'*.npy'))

    for file in files:
        # Load the .npy file
        data = np.load(file)

        # Check if the data is a 2D array
        if len(data.shape) != 2:
            raise ValueError("The .npy file does not contain a 2D array")
    
        # Transpose the data
        data = data.T
        
        # Save the data as a .csv file wihtout the column index and header
        df = pd.DataFrame(data)
        csv_file = file.replace('.npy', '.csv')
        df.to_csv(csv_file, index=False, header=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--npy_file_dir', type=str, required=True)
    args = parser.parse_args()

    npy_to_csv(args.npy_file_dir)