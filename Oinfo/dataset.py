from torch.utils.data import IterableDataset
from itertools import combinations
import math

class LinpartsDataset(IterableDataset):
    def __init__(self, covmat, order):
        self.covmat = covmat
        self.N = self.covmat.shape[0]
        self.order = order
        self.linparts_generator = combinations(range(self.N), self.order)

    def __len__(self):
        """Returns the number of combinations of features of the specified order."""
        return math.comb(self.N, self.order)

    def __iter__(self):
        """
        Iterate over all combinations of features in the dataset.

        Yields:
            tuple: A tuple containing:
                - part (tuple): The indices of the features in the current combination.
                - len_part (int): The number of features in the current combination (always equal to 'order').
                - X_part (np.ndarray): The submatrix of the covariance matrix corresponding to the current combination, shape (order, order).
        """
        for part in self.linparts_generator:
            part = list(part)
            yield part, len(part), self.covmat[part][:,part]
