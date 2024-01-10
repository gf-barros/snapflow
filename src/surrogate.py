import numpy as np


class SVD:
    def __init__(self, data, truncation_basis):
        self.u = None
        self.s = None
        self.vt = None
        self.data = data
        self.truncation_basis = truncation_basis

    def __apply_normalization(self):
        pass

    def __undo_normalization(self):
        pass

    def __svd(self):
        self.u, self.s, self.vt = np.linalg.svd(self.data)
