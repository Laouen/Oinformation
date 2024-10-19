import pandas as pd
from glob import glob
import os


root = '/home/laouen.belloli/Documents/data/Oinfo/fmri_anesthesia/42003_2023_5063_MOESM3_ESM/nets_by_subject'
states_all = ['Awake', 'Deep', 'Mild', 'Recovery']

def read_dataset(root, states):

    dfs_dict = {}
    for state in states_all:
        # list all folder in root/state
        folders = glob(os.path.join(root, state, '*'))
        for folder in folders:
            network = os.path.basename(folder).replace('_parcellation_5', '')
            # list all csv files in folder
            csv_files = glob(os.path.join(folder, f'ts_{network}_parcellation_5_Sub*.csv'))
            for csv_file in csv_files:
                sub = int(os.path.basename(csv_file).split('_')[-1].split('.')[0].replace('Sub', ''))
                
                # Read csv file and add information columns
                df = pd.read_csv(csv_file, sep=',', header=None)
                
                # convert the columns in multilavel, add the network to a second lavel
                df.columns = pd.MultiIndex.from_product([[network], range(df.shape[1])])
                
                # Add df to dfs dict
                if sub not in dfs_dict:
                    dfs_dict[sub] = {}
                
                if state not in dfs_dict[sub]:
                    dfs_dict[sub][state] = []
                
                dfs_dict[sub][state].append(df)

    # Concatenate all dataframes into a single one
    dfs_list = []
    for sub, states in dfs_dict.items():
        for state, dfs in states.items():
            df = pd.concat(dfs, axis=1)
            df['sub'] = sub
            df['state'] = state
            df = df[[('sub',''), ('state','')] + [col for col in df.columns if col[0] not in ['sub', 'state']]]
            dfs_list.append(df)

    return pd.concat(dfs_list, axis=0)
