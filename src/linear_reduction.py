import numpy as np
import matplotlib.pyplot as plt
from src.utils import logger
from pathlib import Path
import os




class SVD:
    """Class responsible for computing the SVD factorization"""

    def __init__(self, data, params_dict, output_folder=None):
        """Instantiation of the SVD class.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        svd_params_dict : dict
            Dictionary containing the SVD parameters.

        Attributes
        ----------
        self.u : numpy.ndarray
            A 2D array containing the left singular vectors modes from self.data.
        self.s : numpy.ndarray
            A 1D array containing the singular values from self.data.
        self.vt : numpy.ndarray
            A 2D array containing the right singular vectors modes from self.data.
        self.data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        self.params_dict : dict
            A dictionary containing all the modeling parameters.
        """
        self.u = None
        self.s = None
        self.vt = None
        self.data = data
        self.svd_params_dict = params_dict["svd"]
        if output_folder:
            svd_output_folder = output_folder / Path("svd")
            if not os.path.exists(svd_output_folder):
                os.mkdir(svd_output_folder)
            self.output_folder = svd_output_folder

    def __diagonalize_s(self):
        """If singular values are in the form of a 1D vector,
        returns a 2D diagonal matrix.
        """
        return np.diag(self.s)

    def __vectorize_s(self):
        """If singular values are in the form of a 2D diagonal matrix,
        returns a 1D vector.
        """
        return np.squeeze(self.s)

    def __truncate_svd(self):
        """Truncates the singular values and vectors using the number of
        vectors specified in the "trunc_basis" key of the SVD parameters.
        """
        self.u = self.u[:, : self.svd_params_dict["trunc_basis"]]
        self.s = self.s[: self.svd_params_dict["trunc_basis"]]
        self.vt = self.vt[:, : self.svd_params_dict["trunc_basis"]]

    def __svd(self):
        """Computes the full SVD using numpy's algorithm (slower)"""
        self.u, self.s, self.vt = np.linalg.svd(self.data, full_matrices=False)
        self.__truncate_svd()

    def __randomized_svd(self):
        """Computes the truncated SVD using rSVD algorithm (faster). Requires
        parameters "trunc_basis", "power_iterations" and "oversampling".
        """
        basis_vectors = self.svd_params_dict.get("trunc_basis")
        power_iterations = self.svd_params_dict.get("power_iterations")
        oversampling = self.svd_params_dict.get("oversampling")
        p_random_vectors = np.random.randn(
            self.data.shape[1], basis_vectors + oversampling
        )
        z_projected_matrix = self.data @ p_random_vectors
        for _ in range(power_iterations):
            z_projected_matrix = self.data @ (self.data.T @ z_projected_matrix)
        q_values, _ = np.linalg.qr(z_projected_matrix, mode="reduced")
        y_reduced_matrix = q_values.T @ self.data
        u_vectors_y, self.s, self.vt = np.linalg.svd(
            y_reduced_matrix, full_matrices=False
        )
        # print(self.u.shape)
        print(q_values.shape)
        print(u_vectors_y.shape)
        self.u = q_values @ u_vectors_y
        self.__truncate_svd()
        return

    def fit(self):
        """Computes the SVD depending on the desired algorithm. Preprocessing steps
        can be applied before factorization.
        """
        match self.svd_params_dict.get("svd_type"):
            case "full_svd":
                self.__svd()
            case "randomized_svd":
                self.__randomized_svd()

    def plot_singular_values(self):
        """Plots singular values computed."""
        plot_data = self.s[:, np.newaxis]
        plot_data = np.insert(plot_data, 1, range(1, plot_data.shape[0] + 1), axis=1)

        # Plotting scatter plot using matplotlib
        fig, ax = plt.subplots()
        ax.scatter(plot_data[:, 1], plot_data[:, 0])

        # Setting x and y axis labels and scales
        ax.set_xlabel("i")
        ax.set_ylabel("Singular Values")
        ax.set_yscale("log")

        if hasattr(self, 'output_folder'):
            plt.savefig(self.output_folder / Path("singular_values.png"))
        plt.close()

