from torch.utils.data import IterableDataset
from itertools import combinations
import math
import numpy as np
import torch

from Oinfo import combinations_range

class CovarianceDataset(IterableDataset):
    def __init__(self, matrix: np.array, partition_order: int):
        self.matrix = torch.tensor(matrix)
        self.n_variables = self.matrix.shape[0]
        self.partition_order = partition_order
        self.partitions_generator = combinations(range(self.n_variables), self.partition_order)

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return math.comb(self.n_variables, self.partition_order)

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



class CovarianceRangeDataset(IterableDataset):
    def __init__(self, matrix: np.ndarray, partition_order: int, start: int, stop: int):
        self.matrix = torch.tensor(matrix)
        self.n_variables = self.matrix.shape[0]
        self.partition_order = partition_order
        self.start = start
        self.stop = stop
        self.partitions_generator = combinations_range(
            self.n_variables, self.partition_order,
            self.start, self.stop
        )
        

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return (self.stop - self.start)

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
