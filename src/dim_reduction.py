import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


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


class AutoEncoder(torch.nn.Module):
    def __init__(self, data, ae_params_dict):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.ae_params_dict = ae_params_dict
        self.encoder = None
        self.decoder = None
        self.__create_nn_graph()

    def __filter_none_layers_from_list(self, list_of_layers):
        return torch.nn.ModuleList(
            list(filter(lambda item: item is not None, list_of_layers))
        )

    def __create_nn_graph(self):
        dict_etc = {
            "linear": torch.nn.Linear,
            "leaky_relu": torch.nn.LeakyReLU,
            "sigmoid": torch.nn.Sigmoid,
            "": lambda x: x,
        }
        input_size = self.data.shape[1]

        layers_encoder = []
        layers_decoder = []

        layers_encoder.append(
            dict_etc["linear"](
                input_size,
                self.ae_params_dict["hidden_layers_sizes"][0],
            )
        )
        layers_encoder.append(
            dict_etc[self.ae_params_dict["hidden_layers_activation_function"][0]](
                self.ae_params_dict["hidden_layers_activation_function_parameters"][0]
            )
        )

        layers_decoder.append(
            dict_etc[self.ae_params_dict["decoder_activation_function"]]()
        )

        layers_decoder.append(
            dict_etc["linear"](
                self.ae_params_dict["hidden_layers_sizes"][0],
                input_size,
            )
        )

        for layer_number in range(1, self.ae_params_dict["number_of_hidden_layers"]):
            layers_encoder.append(
                dict_etc["linear"](
                    self.ae_params_dict["hidden_layers_sizes"][layer_number - 1],
                    self.ae_params_dict["hidden_layers_sizes"][layer_number],
                )
            )
            layers_encoder.append(
                dict_etc[
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
                dict_etc[
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
                dict_etc["linear"](
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
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderTrain:
    def __init__(self, data, ae_params_dict):
        self.auto_encoder = AutoEncoder(data, ae_params_dict)
        self.data_loader = None
        self.data_loader_training = None
        self.outputs = {}
        self.data = data
        self.ae_params_dict = ae_params_dict

    def __load_data(self, key_batch_size, key_num_workers, shuffle_flag):
        data_loader = DataLoader(
            dataset=self.data,
            batch_size=self.ae_params_dict[key_batch_size],
            num_workers=self.ae_params_dict[key_num_workers],
            shuffle=shuffle_flag,
        )
        return data_loader

    def __init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )

    def __item_function(self, x):
        return x.item()

    def fit(self):
        self.auto_encoder.apply(self.__init_weights)
        self.data_loader = self.__load_data("batch_size", "num_workers", True)
        loss_function = torch.nn.SmoothL1Loss(beta=2.0)
        optimizer = torch.optim.Adam(
            self.auto_encoder.parameters(),
            lr=1e-4,  # TODO: parametrize
            weight_decay=1e-8,  # TODO: parametrize
        )

        epochs = self.ae_params_dict["num_epochs"]

        self.outputs["outputs"] = []
        self.outputs["avg_loss_by_epoch"] = []

        for _ in tqdm(range(epochs)):
            losses_batches_per_epoch = []
            for _, image in enumerate(self.data_loader):
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
