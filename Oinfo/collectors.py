
import pandas as pd
import numpy as np

def save_to_csv(output_path, N, nplet_tc, nplet_dtc, nplet_o, nplet_s, partition_idxs, sep='\t', only_synergestic=False):

    # if only_synergestic; remove nplets with nplet_o >= 0
    if only_synergestic:
        print('Removing non synergetic values')
        to_keep = np.where(nplet_o < 0)[0]
        nplet_tc = nplet_tc[to_keep]
        nplet_dtc = nplet_dtc[to_keep]
        nplet_o = nplet_o[to_keep]
        nplet_s = nplet_s[to_keep]
        partition_idxs = [partition_idxs[i] for i in to_keep]

    # Create dataframe with the measurements
    df_meas = pd.DataFrame({
        'tc': nplet_tc,
        'dtc': nplet_dtc,
        'o': nplet_o,
        's': nplet_s
    })

    # Create a DataFrame with the n-plets
    batch_size, order = partition_idxs.shape
    bool_array = np.zeros((batch_size, N), dtype=bool)
    rows = np.arange(batch_size).reshape(-1, 1)
    bool_array[rows, partition_idxs] = True
    df_vars = pd.DataFrame(bool_array, columns=[f'variable_{i}' for i in range(N)])

    # Concat both dataframes columns and store in disk
    df = pd.concat([df_meas, df_vars], axis=1)
    df.to_csv(output_path, index=False, sep=sep)