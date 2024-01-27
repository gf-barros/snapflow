import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import map_input_function_pytorch
import logging
from pathlib import Path
import h5py
import os

# Disables log messages when using matplotlib
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.ticker").disabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AutoEncoderCreator(torch.nn.Module):
    """Class responsible for creating the AutoEncoder NN."""

    def __init__(self, data, params_dict):
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
        self.params_dict : dict
            A dictionary containing all the modeling parameters.
        self.encoder : torch.nn.modules.container.Sequential
            Neural Network architecture for the encoder.
        self.decoder : torch.nn.modules.container.Sequential
            Neural Network architecture for the decoder.
        """
        super().__init__()
        self.data = torch.tensor(data.T, dtype=torch.float32)
        self.ae_params_dict = params_dict["auto_encoder"]
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

    def __init__(self, data, params_dict, output_folder=None):
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
        self.params_dict : dict
            A dictionary containing all the modeling parameters.
        """
        self.auto_encoder = AutoEncoderCreator(data, params_dict)
        self.data_loader = None
        self.data_loader_training = None
        self.normalization_technique_class = None
        self.outputs = {}
        self.data = self.auto_encoder.data
        self.ae_params_dict = params_dict["auto_encoder"]
        if output_folder:
            ae_output_folder = output_folder / Path("auto_encoder")
            os.mkdir(ae_output_folder)
            self.output_folder = ae_output_folder

    def __train_test_split(self):
        pass  # TODO:

    def __log_run(self, log_type="training"):
        if log_type == "training":
            logger.info(
                "-------------------- Starting training for AutoEncoder --------------------"
            )
            logger.info(f" # of data: {self.data.shape}")
            logger.info(
                f" # of layers: {self.ae_params_dict['number_of_hidden_layers']}"
            )
            logger.info(f" Encoder architecture: {self.auto_encoder.encoder}")

    def __load_data(self, data, key_batch_size, key_num_workers, shuffle_flag):
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
            dataset=data,
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
        self.__log_run("training")
        self.auto_encoder.apply(self.__init_weights)
        self.data_loader = self.__load_data(self.data, "batch_size", "num_workers", True)
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
            for entry in self.data_loader:
                #entry = entry.to(torch.float32)
                optimizer.zero_grad()
                reconstructed = self.auto_encoder(entry)
                loss = loss_function(reconstructed, entry)
                loss.backward()
                optimizer.step()

                losses_batches_per_epoch.append(loss)
            temp_loss_batch = losses_batches_per_epoch
            l_loss = np.array(list(map(self.__item_function, temp_loss_batch)))
            temp_avg_loss_epoch = l_loss.mean()
            self.outputs["avg_loss_by_epoch"].append(temp_avg_loss_epoch)


    def predict(self, forward_data):
        """Predicts data after trained AutoEncoder"""
        self.__log_run("predicting")
        with torch.no_grad():
            input_tensor = torch.from_numpy(forward_data.T).float()
            predictions = self.auto_encoder(input_tensor)
        return predictions.numpy().T

    def encode(self):
        """After training, encodes data for surrogate modeling"""
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

    def plot_quantities_per_epoch(self, quantity, save_only=True):
        """Plots quantities computed per epoch."""
        plot_data = np.array(self.outputs[quantity])
        plot_data = plot_data[:, np.newaxis]
        plot_data = np.insert(plot_data, 1, range(1, plot_data.shape[0] + 1), axis=1)

        fig, ax = plt.subplots()
        ax.scatter(plot_data[:, 1], plot_data[:, 0])
        ax.set_xlabel("epochs")
        ax.set_ylabel(quantity)
        ax.set_yscale("log")

        if hasattr(self, 'output_folder'):
            plt.savefig(self.output_folder / Path(f"{quantity}.png"))

        if not save_only:
            plt.show()

