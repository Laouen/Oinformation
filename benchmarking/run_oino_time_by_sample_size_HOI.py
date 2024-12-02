import pandas as pd
import numpy as np
import os
import time
from glob import glob
from tqdm import tqdm
import hoi

from argparse import ArgumentParser

def main(files_dir: str, output_path: str):

    # get all files
    files = glob(os.path.join(files_dir, f"nsamples-*_nvars-10.csv"))

    nplet = [np.arange(10).tolist()]

    rows = []
    pbar = tqdm(files)
    for file in pbar:
        X = pd.read_csv(file, header=None).values
        T, N = X.shape
        pbar.set_description(f"T={T}:")

        for i in range(100):

            start_time = time.time()
            h = hoi.metrics.Oinfo(X)
            h.fit(10, 10, method='gc')
            delta_t = time.time() - start_time

            rows.append([T, i, delta_t])
    
        pd.DataFrame(rows, columns=['sample size', 'iteration', 'time']).to_csv(output_path, sep='\t', index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--files_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args.files_dir, args.output_path)