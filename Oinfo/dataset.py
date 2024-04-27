from torch.utils.data import IterableDataset
from itertools import combinations
import math
import numpy as np
import torch

class MatrixPartitionDataset(IterableDataset):
    def __init__(self, matrix: np.array, n_variables: int, partition_order: int):
        self.matrix = torch.tensor(matrix)
        self.n_variables = n_variables
        self.partition_order = partition_order
        self.partitions_generator = combinations(range(self.n_variables), self.partition_order)

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return math.comb(self.n_variables, self.partition_order)

class CovarianceDataset(MatrixPartitionDataset):
    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                - partition_idxs (list): The indices of the features in the current combination.
                - partition_covmat (np.ndarray): The submatrix of the covariance matrix corresponding to the current combination, shape (order, order).
        """
        for partition_idxs in self.partitions_generator:
            partition_idxs = np.array(partition_idxs)

            # (order, order)
            yield partition_idxs, self.matrix[partition_idxs][:,partition_idxs]


class KNearestNeighborDataset(MatrixPartitionDataset):
    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                - partition_idxs (list): The indices of the features in the current combination.
                - sub_distances (np.ndarray): The submatrix with the distances to the K-th nearest neighbor from the current combination, shape (order, 1).
        """
        for partition_idxs in self.partitions_generator:
            partition_idxs = np.array(partition_idxs)

            # (n_samples, n_samples, order)
            yield partition_idxs, self.matrix[:, :, partition_idxs]