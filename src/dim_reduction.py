import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .utils import map_input_function_pytorch


class SVD:
    """Class responsible for computing the SVD factorization"""

    def __init__(self, data, svd_params_dict):
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
        self.svd_params_dict : dict
            A dictionary containing the SVD parameters
        """
        self.u = None
        self.s = None
        self.vt = None
        self.data = data
        self.svd_params_dict = svd_params_dict

    def __apply_normalization(self):
        pass

    def __undo_normalization(self):
        pass

    def diagonalize_s(self):
        """If singular values are in the form of a 1D vector,
        returns a 2D diagonal matrix.
        """
        return np.diag(self.s)

    def vectorize_s(self):
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
        u_vectors_y, self.s, self.vt = np.linalg.svd(y_reduced_matrix, full_matrices=0)
        self.u = q_values @ u_vectors_y
        self.__truncate_svd()
        return

    def fit(self):
        """Computes the SVD depending on the desired algorithm. Preprocessing steps
        can be applied before factorization.
        """
        match self.svd_params_dict.get("preprocessing_type"):
            case "min_max":
                pass  # TODO
            case "mean_subtraction":
                pass  # TODO
            case None:
                pass
        match self.svd_params_dict.get("svd_type"):
            case "full_svd":
                self.__svd()
            case "randomized_svd":
                self.__randomized_svd()


class AutoEncoderCreator(torch.nn.Module):
    """Class responsible for creating the AutoEncoder NN."""

    def __init__(self, data, ae_params_dict):
        """Instantiation of the AutoEncoderCreator class.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        ae_params_dict : dict
            Dictionary containing the AutoEncoder parameters.

        Attributes
        ----------
        self.data : numpy.ndarray
            A 2D array containing the stacked snapshots. Eventually converted to
            torch.tensor.
        self.ae_params_dict : dict
            A dictionary containing the AutoEncoder parameters
        self.encoder : torch.nn.modules.container.Sequential
            Neural Network architecture for the encoder.
        self.decoder : torch.nn.modules.container.Sequential
            Neural Network architecture for the decoder.
        """
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.ae_params_dict = ae_params_dict
        self.encoder = None
        self.decoder = None
        self.__create_nn_graph()

    def __filter_none_layers_from_list(self, list_of_layers):
        """Filters None from list layers (situations where no
        activation function is needed)

        Parameters
        ----------
        list_of_layers : list
            List of layers for the encoder or decoder.

        Returns
        -------
        torch.nn.ModuleList
            ModuleList containing the filtered layers.
        """
        return torch.nn.ModuleList(
            list(
                filter(
                    lambda item: (item is not None) and (item != "None"),
                    list_of_layers,
                )
            )
        )

    def __create_nn_graph(self):
        """Creates self.decoder and self.encoder from YAML file."""
        input_size = self.data.shape[1]

        layers_encoder = []
        layers_decoder = []

        # Defining first layers for the encoder
        layers_encoder.append(
            map_input_function_pytorch["linear"](
                input_size,
                self.ae_params_dict["hidden_layers_sizes"][0],
            )
        )
        layers_encoder.append(
            map_input_function_pytorch[
                self.ae_params_dict["hidden_layers_activation_function"][0]
            ](self.ae_params_dict["hidden_layers_activation_function_parameters"][0])
        )

        # Defining first layers for the decoder
        layers_decoder.append(
            map_input_function_pytorch[
                self.ae_params_dict["decoder_activation_function"]
            ]()
        )

        layers_decoder.append(
            map_input_function_pytorch["linear"](
                self.ae_params_dict["hidden_layers_sizes"][0],
                input_size,
            )
        )

        for layer_number in range(1, self.ae_params_dict["number_of_hidden_layers"]):
            layers_encoder.append(
                map_input_function_pytorch["linear"](
                    self.ae_params_dict["hidden_layers_sizes"][layer_number - 1],
                    self.ae_params_dict["hidden_layers_sizes"][layer_number],
                )
            )
            layers_encoder.append(
                map_input_function_pytorch[
                    self.ae_params_dict["hidden_layers_activation_function"][
                        layer_number
                    ]
                ](
                    self.ae_params_dict["hidden_layers_activation_function_parameters"][
                        layer_number
                    ]
                )
            )
            layers_decoder.append(
                map_input_function_pytorch[
                    self.ae_params_dict["hidden_layers_activation_function"][
                        layer_number - 1
                    ]
                ](
                    self.ae_params_dict["hidden_layers_activation_function_parameters"][
                        layer_number - 1
                    ]
                )
            )
            layers_decoder.append(
                map_input_function_pytorch["linear"](
                    self.ae_params_dict["hidden_layers_sizes"][layer_number],
                    self.ae_params_dict["hidden_layers_sizes"][layer_number - 1],
                )
            )

        layers_encoder = self.__filter_none_layers_from_list(layers_encoder)
        layers_decoder = self.__filter_none_layers_from_list(layers_decoder)

        self.encoder = torch.nn.Sequential(*torch.nn.ModuleList(layers_encoder))
        self.decoder = torch.nn.Sequential(
            *torch.nn.ModuleList(reversed(layers_decoder))
        )

    def forward(self, data):
        """forward pass for the AutoEncoder architecture."""
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder:
    """The AutoEncoder Class, responsible for instantiating the AutoEncoderCreator class and
    NN training"""

    def __init__(self, data, ae_params_dict):
        """Instantiation of the AutoEncoder class.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        ae_params_dict : dict
            Dictionary containing the AutoEncoder parameters.

        Attributes
        ----------
        self.auto_encoder : AutoEncoderCreator object
            Neural Network architecture creator class for the AutoEncoder.
        self.data_loader : torch.utils.data.dataloader.DataLoader
            DataLoader for NN training.
        self.data_loader_training: ????
            ?????
        self.outputs : dict
            Dictionary containing the training outputs.
        self.data : numpy.ndarray
            A 2D array containing the stacked snapshots. Eventually converted to
            torch.tensor.
        self.ae_params_dict : dict
            A dictionary containing the AutoEncoder parameters
        """
        self.auto_encoder = AutoEncoderCreator(data, ae_params_dict)
        self.data_loader = None
        self.data_loader_training = None
        self.outputs = {}
        self.data = data
        self.ae_params_dict = ae_params_dict

    def __load_data(self, key_batch_size, key_num_workers, shuffle_flag):
        """Sets up DataLoader

        Parameters
        ----------
        key_batch_size : int
            _description_
        key_num_workers : int
            _description_
        shuffle_flag : bool
            _description_

        Returns
        -------
        torch.utils.data.dataloader.DataLoader
            _description_
        """
        data_loader = DataLoader(
            dataset=self.data,
            batch_size=self.ae_params_dict[key_batch_size],
            num_workers=self.ae_params_dict[key_num_workers],
            shuffle=shuffle_flag,
        )
        return data_loader

    def __init_weights(self, m):
        """Initializes weights according to desired strategy.

        Parameters
        ----------
        m : _type_
            _description_
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

    def __item_function(self, x):
        return x.item()

    def fit(self):
        """Trains AutoEncoder"""
        self.auto_encoder.apply(self.__init_weights)
        self.data_loader = self.__load_data("batch_size", "num_workers", True)
        loss_function = map_input_function_pytorch[
            self.ae_params_dict["loss_function"]
        ](self.ae_params_dict["loss_parameters"]["beta"])
        print(type(self.data_loader))
        optimizer = torch.optim.Adam(
            self.auto_encoder.parameters(),
            lr=float(self.ae_params_dict["learning_rate"]),
            weight_decay=float(self.ae_params_dict["weight_decay"]),
        )

        epochs = self.ae_params_dict["num_epochs"]

        self.outputs["outputs"] = []
        self.outputs["avg_loss_by_epoch"] = []

        for _ in tqdm(range(epochs)):
            losses_batches_per_epoch = []
            for __, image in enumerate(self.data_loader):
                image = image.to(torch.float32)
                reconstructed = self.auto_encoder(image)
                loss = loss_function(reconstructed, image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_batches_per_epoch.append(loss)
            temp_loss_batch = losses_batches_per_epoch
            l_loss = np.array(list(map(self.__item_function, temp_loss_batch)))
            temp_avg_loss_epoch = l_loss.mean()
            self.outputs["avg_loss_by_epoch"].append(temp_avg_loss_epoch)
            self.outputs["outputs"].append((epochs, image, reconstructed))

    def compute_training_error(self):
        """???"""
        print("\nCalculating training error for each vector")
        self.data_loader_training = self.__load_data(
            "batch_size_training_error", "num_workers_training_errors", False
        )
        loss_function = torch.nn.SmoothL1Loss(beta=2.0)

        self.outputs["error_training"] = []
        self.outputs["reconstructed_vectors"] = []

        for i, vector in enumerate(loader_2):
            encoder_decoder = self.auto_encoder(vector)
            error = loss_function(vector, encoder_decoder)
            self.outputs["error_training"].append(error)

            self.outputs["reconstructed_vectors"].append(encoder_decoder)

        self.outputs["error_training_np"] = np.array(
            list(map(self.__item_function, self.outputs["error_training"]))
        )
