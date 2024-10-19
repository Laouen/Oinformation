import pandas as pd
import numpy as np
import os
import time
from glob import glob
from argparse import ArgumentParser
from thoi.commons import gaussian_copula
from tqdm import tqdm

from thoi.measures.gaussian_copula import nplets_measures

def main(files_dir: str, output_path: str):

    # get all files
    files = glob(os.path.join(files_dir, f"nsamples-*_nvars-10.csv"))

    nplet = np.arange(10)

    rows = []
    pbar = tqdm(files)
    for file in pbar:
        X = pd.read_csv(file, header=None).values
        T, N = X.shape
        pbar.set_description(f"T={T}:")

        for i in range(100):

            start_time = time.time()
            gaussian_copula(X)
            gc_time = time.time() - start_time
    
            start_time = time.time()
            nplets_measures(X, nplet, use_cpu=True)
            nplets_time = time.time() - start_time

            rows.append([T, i, gc_time, nplets_time])
    
    pd.DataFrame(rows, columns=['sample size', 'iteration', 'gaussion copula time', 'O-informtion time']).to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--files_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(args.files_dir, args.output_path)