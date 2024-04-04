from torch.utils.data import IterableDataset
from itertools import combinations
import numpy as np

class LinpartsDataset(IterableDataset):
    def __init__(self, covmat, N, order):
        self.N = N
        self.order = order
        self.covmat = covmat
        self.linparts_generator = combinations(range(self.N), self.order)

    def __len__(self):
        n_fact = np.math.factorial(self.N)
        r_fact = np.math.factorial(self.order)
        n_r_fact = np.math.factorial(self.N - self.order)
        return int(n_fact / (r_fact * n_r_fact))

    def __iter__(self):
        for part in self.linparts_generator:
            yield part, len(part), self.covmat[part][:,part]
