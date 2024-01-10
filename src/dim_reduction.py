import numpy as np


class SVD:
    def __init__(self, data, svd_params_dict):
        self.u = None
        self.s = None
        self.vt = None
        self.data = data
        self.svd_params_dict = svd_params_dict

    def __apply_normalization(self):
        pass

    def __undo_normalization(self):
        pass

    def __diagonalize_s(self):
        return np.diag(self.s)

    def __vectorize_s(self):
        return np.squeeze(self.s)

    def __truncate_svd(self):
        pass  # TODO: implement svd truncation SVD

    def __svd(self):
        self.u, self.s, self.vt = np.linalg.svd(self.data, full_matrices=False)
        self.__truncate_svd()

    def __randomized_svd(self):
        basis_vectors = self.svd_params_dict["trunc_basis"]
        power_iterations = self.svd_params_dict["power_iterations"]
        oversampling = self.svd_params_dict["oversampling"]
        p_random_vectors = np.random.randn(
            self.data.shape[1], basis_vectors + oversampling
        )
        z_projected_matrix = self.data @ p_random_vectors
        for _ in range(power_iterations):
            z_projected_matrix = self.data @ (self.data.T @ z_projected_matrix)
        q_values, _ = np.linalg.qr(z_projected_matrix, mode="reduced")
        y_reduced_matrix = q_values.T @ self.data
        u_vectors_y, self.s, self.vt = np.linalg.svd(y_reduced_matrix, full_matrices=0)
        self.u = q_values @ u_vectors_y
        self.__truncate_svd()
        return

    def fit(self):
        match self.svd_params_dict.get("preprocessing_type"):
            case "min_max":
                pass  # TODO
            case "mean_subtraction":
                pass  # TODO
            case None:
                print("b")
        match self.svd_params_dict.get("svd_type"):
            case "full_svd":
                self.__svd()
            case "randomized_svd":
                self.__randomized_svd()
