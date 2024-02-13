import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from snapflow.utils import map_input_function_pytorch, check_parameters_and_extract, check_parameters_and_extract_layers
from snapflow.utils import logger, timing_decorator
from pathlib import Path
import h5py
import os





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
    
    def __create_linear_layer(self, input_size, output_size):
        """Creates a linear layer."""
        return map_input_function_pytorch["linear"](input_size, output_size)

    def __create_activation_layer(self, key, layer=0):
        """Creates an activation layer based on the specified activation function and parameters."""
        return check_parameters_and_extract_layers(self.ae_params_dict, key, layer)

    def __construct_encoder_layers(self):
        """Constructs all layers for the encoder."""
        layers = []
        input_size = self.data.shape[1]

        # Add first encoder layer
        layers.append(self.__create_linear_layer(input_size, self.ae_params_dict["hidden_layers_sizes"][0]))

        # Add subsequent layers
        for layer_number in range(1, self.ae_params_dict["number_of_hidden_layers"]):
            layers.append(self.__create_activation_layer("hidden_layers_activation_function", layer_number - 1))
            layers.append(self.__create_linear_layer(self.ae_params_dict["hidden_layers_sizes"][layer_number - 1],
                                                     self.ae_params_dict["hidden_layers_sizes"][layer_number]))
        
        return layers

    def __construct_decoder_layers(self):
        """Constructs all layers for the decoder."""
        layers = []


        # Add all layers in reverse order
        for layer_number in reversed(range(self.ae_params_dict["number_of_hidden_layers"] - 1)):
            layers.append(self.__create_linear_layer(self.ae_params_dict["hidden_layers_sizes"][layer_number + 1],
                                                     self.ae_params_dict["hidden_layers_sizes"][layer_number ]))
            layers.append(self.__create_activation_layer("hidden_layers_activation_function", layer_number))



        # Add the final activation layer and linear layer to map back to the input size
        input_size = self.data.shape[1]
        layers.append(self.__create_linear_layer(self.ae_params_dict["hidden_layers_sizes"][0], input_size))
        layers.append(self.__create_activation_layer("decoder_activation_function"))

        return layers

    def __create_nn_graph(self):
        """Creates self.decoder and self.encoder from YAML file."""
        encoder_layers = self.__construct_encoder_layers()
        decoder_layers = self.__construct_decoder_layers()

        encoder_layers = self.__filter_none_layers_from_list(encoder_layers)
        decoder_layers = self.__filter_none_layers_from_list(decoder_layers)

        self.encoder = torch.nn.Sequential(*torch.nn.ModuleList(encoder_layers))
        self.decoder = torch.nn.Sequential(*torch.nn.ModuleList(decoder_layers))
    
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device used: {device}")
        if output_folder:
            ae_output_folder = output_folder / Path("auto_encoder")
            if not os.path.exists(ae_output_folder):
                os.mkdir(ae_output_folder)
            self.output_folder = ae_output_folder

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
            logger.info(f" Decoder architecture: {self.auto_encoder.decoder}")


    def __init_weights(self, m):
        """Initializes weights according to desired strategy.

        Parameters
        ----------
        m : _type_
            _description_
        """
        if isinstance(m, torch.nn.Linear):
            check_parameters_and_extract(self.ae_params_dict, "initializer", m.weight)

    def __item_function(self, x):
        return x.item()

    @timing_decorator
    def fit(self):
        """Trains AutoEncoder"""
        self.__log_run("training")
        self.auto_encoder.apply(self.__init_weights)
        
        self.data_loader = check_parameters_and_extract(self.ae_params_dict, "data_loader", extra_param=self.data)
        loss_function = check_parameters_and_extract(self.ae_params_dict, "loss_function")
        optimizer = check_parameters_and_extract(self.ae_params_dict, "optimizer", extra_param=self.auto_encoder.parameters())

        epochs = self.ae_params_dict["num_epochs"]

        self.outputs["outputs"] = []
        self.outputs["avg_loss_by_epoch"] = []

        for _ in tqdm(range(epochs)):
            losses_batches_per_epoch = []
            for entry in self.data_loader:
                # entry = entry.to(torch.float32)
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

    @timing_decorator
    def predict(self, forward_data):
        """Predicts data after trained AutoEncoder"""
        self.__log_run("predicting")
        with torch.no_grad():
            input_tensor = torch.from_numpy(forward_data.T).float()
            predictions = self.auto_encoder(input_tensor)
        return predictions.numpy().T

    def encode(self, data):
        """After training, encodes data for surrogate modeling"""
        with torch.no_grad():
            input_tensor = torch.from_numpy(data.T).float()
            encoded = self.auto_encoder.encoder(input_tensor)
        return encoded.detach().numpy()


    def plot_quantities_per_epoch(self, quantity, fold=0):
        """Plots quantities computed per epoch."""
        plot_data = np.array(self.outputs[quantity])
        plot_data = plot_data[:, np.newaxis]
        plot_data = np.insert(plot_data, 1, range(1, plot_data.shape[0] + 1), axis=1)

        fig, ax = plt.subplots()
        ax.scatter(plot_data[:, 1], plot_data[:, 0])
        ax.set_xlabel("epochs")
        ax.set_ylabel(quantity)
        ax.set_yscale("log")

        if hasattr(self, "output_folder"):
            plt.savefig(self.output_folder / Path(f"{quantity}_{fold}.png"))
        plt.close()
