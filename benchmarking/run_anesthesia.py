from glob import glob
import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import wilcoxon

from thoi.heuristics.simulated_annealing_multi_order import simulated_annealing_multi_order
from thoi.heuristics import greedy

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import time

def wilcoxon_metric(batched_res: torch.Tensor, metric:str='o'):
    
    '''
    Get the metric from the batched results returning the average over the D axis.

    params:
    - batched_res (np.ndarray): The batched results with shape (batch_size, D, 4) where 4 is the number of metrics (tc, dtc, o, s). D = 2*n_subjects where [0, D/2) are from the group 1 and [D/2, D) are from the group 2.
    - metric (str): The metric to test the difference. One of tc, dtc, o or s
    '''
    
    METRICS = ['tc', 'dtc', 'o', 's']
    metric_idx = METRICS.index(metric)
    
    batch_size, D = batched_res.shape[:2]
    
    # |batch_size| x |D/2|
    group1 = batched_res[:, :D//2, metric_idx]
    group2 = batched_res[:, D//2:, metric_idx]
    
    # for each batch item compute the wilcoxon test
    # |batch_size|
    pvals = torch.tensor([
        wilcoxon(group1[i].cpu().numpy(), group2[i].cpu().numpy(), alternative='two-sided').pvalue
        for i in range(batch_size)
    ])
    
    return pvals

def auc_metric(batched_res: torch.Tensor, metric:str='o'):
    
    '''
    Get the metric from the batched results returning the average over the D axis.

    params:
    - batched_res (np.ndarray): The batched results with shape (batch_size, D, 4) where 4 is the number of metrics (tc, dtc, o, s). D = 2*n_subjects where [0, D/2) are from the group 1 and [D/2, D) are from the group 2.
    - metric (str): The metric to test the difference. One of tc, dtc, o or s
    '''
    
    METRICS = ['tc', 'dtc', 'o', 's']
    metric_idx = METRICS.index(metric)

    n_subjects = batched_res.shape[1] // 2    
    
    # Prepare de data    
    batched_X = batched_res[..., metric_idx].cpu().numpy()
    y = np.concatenate([np.zeros(n_subjects), np.ones(n_subjects)])
    
    # Prepare the output
    return torch.tensor([roc_auc_score(y, X) for X in batched_X])

def cross_val_ml_metric(batched_res: torch.Tensor, metric:str='o', classifier=None):
    
    '''
    Get the metric from the batched results returning the average over the D axis.

    params:
    - batched_res (np.ndarray): The batched results with shape (batch_size, D, 4) where 4 is the number of metrics (tc, dtc, o, s). D = 2*n_subjects where [0, D/2) are from the group 1 and [D/2, D) are from the group 2.
    - metric (str): The metric to test the difference. One of tc, dtc, o or s
    '''
    
    if classifier is None:
        classifier = LogisticRegression()
    
    t = time.time()
    METRICS = ['tc', 'dtc', 'o', 's']
    metric_idx = METRICS.index(metric)

    n_subjects = batched_res.shape[1] // 2    
    
    # Prepare de data    
    batched_X = [x for x in batched_res[..., metric_idx].cpu().numpy()]
    groups = np.concatenate([np.arange(n_subjects),np.arange(n_subjects)])
    y = np.concatenate([np.zeros(n_subjects), np.ones(n_subjects)])
    
    # Prepare the classifier
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])
    
    # Compute the leave one out cross validation roc auc score
    auc_scores = []
    for X in batched_X:
        # Compute the leave one out cross validation roc auc score
        logo = LeaveOneGroupOut()
        y_prob = np.zeros_like(y)
        for train_idx, test_idx in logo.split(X, y, groups):
            clf.fit(X[train_idx].reshape(-1, 1), y[train_idx])
            y_prob[test_idx] = clf.predict_proba(X[test_idx].reshape(-1, 1))[:,1]
        
        auc_scores.append(roc_auc_score(y, y_prob))
        
    total_time = time.time() - t

    return torch.tensor(auc_scores)

def read(data_dir: str):
    
    #states_all = ['Awake', 'Deep', 'Mild', 'Recovery']
    states_all = ['Awake', 'Deep']

    dfs_dict = {}
    for state in states_all:
        # list all folder in data_dir/state
        folders = glob(os.path.join(data_dir, state, '*'))
        for folder in folders:
            network = os.path.basename(folder).replace('_parcellation_5', '')
            
            # List all csv files in folder
            csv_files = glob(os.path.join(folder, f'ts_{network}_parcellation_5_Sub*.csv'))
            for csv_file in csv_files:
                sub = int(os.path.basename(csv_file).split('_')[-1].split('.')[0].replace('Sub', ''))
                
                # Read csv file and add information columns
                df = pd.read_csv(csv_file, sep=',', header=None)
                
                # Convert the columns in multilavel, add the network to a second lavel
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

    df = pd.concat(dfs_list, axis=0)

    # Remove sub10 that has no some missing networks
    df = df[df['sub'] != 10]

    # Check Show missing values (should be none), all subjects have all the networks
    assert df.isnull().sum().sum() == 0, 'There are missing values'

    # Check all subjects have same lenght across states and networks
    assert df.groupby(['sub','state']).size().unique() == [245], 'Series lenght is not the same for all subjects'

    networks = df.columns.get_level_values(0).unique()[2:]
    #network_pairs = list(combinations(networks, 2))

    # Create the list of datsa
    Xs = [
        df[(df['sub'] == sub) & (df['state'] == state)][networks].values
        for state in states_all
        for sub in sorted(df['sub'].unique())
    ]

    print(networks)

    return df, Xs

def run_greedy_roc_auc(Xs, root):
    nplets, scores = greedy(Xs, repeat=50, metric=auc_metric, largest=True)
    
    # save the results as npy files
    np.save(os.path.join(root, 'nplets_greedy_roc_auc.npy'), nplets.numpy())
    np.save(os.path.join(root, 'scores_greedy_roc_auc.npy'), scores.numpy())

def run_annealing_roc_auc(Xs, root):
    nplets, scores = simulated_annealing_multi_order(Xs, repeat=50, metric=auc_metric, largest=True)
    
    # save the results as npy files
    np.save(os.path.join(root, 'nplets_annealing_roc_auc.npy'), nplets.numpy())
    np.save(os.path.join(root, 'scores_annealing_roc_auc.npy'), scores.numpy())
    
def run_wale(Xs, root):
    Xs_awake = Xs[:16]
    Xs_deep = Xs[16:]
    
    nplets_awake_max, scores_awake_max = greedy(Xs_awake, repeat=50, metric='o', largest=True)
    nplets_awake_min, scores_awake_min = greedy(Xs_awake, repeat=50, metric='o', largest=False)

    nplets_deep_max, scores_deep_max = greedy(Xs_deep, repeat=50, metric='o', largest=True)
    nplets_deep_min, scores_deep_min = greedy(Xs_deep, repeat=50, metric='o', largest=False)
    

    # save the results as npy files
    np.save(os.path.join(root, 'nplets_greedy_awake_max.npy'), nplets_awake_max.numpy())
    np.save(os.path.join(root, 'nplets_greedy_awake_min.npy'), nplets_awake_min.numpy())
    
    np.save(os.path.join(root, 'scores_greedy_awake_max.npy'), scores_awake_max.numpy())
    np.save(os.path.join(root, 'scores_greedy_awake_min.npy'), scores_awake_min.numpy())
    
    np.save(os.path.join(root, 'nplets_greedy_deep_max.npy'), nplets_deep_max.numpy())
    np.save(os.path.join(root, 'nplets_greedy_deep_min.npy'), nplets_deep_min.numpy())
    
    np.save(os.path.join(root, 'scores_greedy_deep_max.npy'), scores_deep_max.numpy())
    np.save(os.path.join(root, 'scores_greedy_deep_min.npy'), scores_deep_min.numpy())
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--input_path', type=str)
    
    args = parser.parse_args()
    
    df, Xs = read(args.input_path)
    
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    
    run_greedy_roc_auc(Xs, args.output_path)
    run_annealing_roc_auc(Xs, args.output_path)
    run_wale(Xs, args.output_path)
    
    print('FINISHED')