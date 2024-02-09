import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils import map_input_function_pytorch, check_parameters_and_extract
from src.utils import logger, timing_decorator
from pathlib import Path
import h5py
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetworkCreator(torch.nn.Module):
    """Class responsible for creating the NN."""

    def __init__(self, input_data, target_data, params_dict):
        """Instantiation of the NeuralNetworkCreator class.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        params_dict : dict
            Dictionary containing the overall parameters.

        Attributes
        ----------
        self.data : numpy.ndarray
            A 2D array containing the stacked snapshots. Eventually converted to
            torch.tensor.
        self.nn : torch.nn.modules.container.Sequential
            Neural Network architecture for the encoder.
        self.nn_params_dict : dict
            A dictionary containing all the modeling parameters.
        """
        super().__init__()
        self.input_data = torch.tensor(input_data.T, dtype=torch.float32)
        self.target_data = torch.tensor(target_data.T, dtype=torch.float32)        
        self.nn_params_dict = params_dict["neural_network"]
        self.nn = None
        self.layers_nn = []
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
    
    def __create_nn_layer(self, layer_input_size, layer_output_size, layer_number=0):
        self.layers_nn.append(
            map_input_function_pytorch["linear"](
                layer_input_size,
                layer_output_size,
            )
        )
        if self.nn_params_dict["hidden_layers_activation_function_parameters"]["active"][layer_number]: 
            self.layers_nn.append(
                map_input_function_pytorch[
                    self.nn_params_dict["hidden_layers_activation_function"][layer_number]
                ](**self.nn_params_dict["hidden_layers_activation_function_parameters"]["parameters"][layer_number])
            )
        else: 
            self.layers_nn.append(
                map_input_function_pytorch[
                    self.nn_params_dict["hidden_layers_activation_function"][layer_number]
                ]()
            )


    def __create_nn_graph(self):
        """Creates self.nn from YAML file."""
        input_size = self.input_data.shape[1]
        target_size = self.target_data.shape[1]

        # Defining first layer for the NN
        self.__create_nn_layer(input_size, self.nn_params_dict["hidden_layers_sizes"][0])

        for layer_number in range(1, self.nn_params_dict["number_of_hidden_layers"]):
            self.__create_nn_layer(self.nn_params_dict["hidden_layers_sizes"][layer_number - 1], 
                                   self.nn_params_dict["hidden_layers_sizes"][layer_number],
                                   layer_number)
        

        self.layers_nn = self.__filter_none_layers_from_list(self.layers_nn)

        self.nn = torch.nn.Sequential(*torch.nn.ModuleList(self.layers_nn))

    def forward(self, data):
        """forward pass for the NN architecture."""
        forward_pass = self.nn(data)
        return forward_pass


class NeuralNetwork:
    """The NeuralNetwork Class, responsible for instantiating the NeuralNetworkCreator class and
    NN training"""

    def __init__(self, input_data, target_data, params_dict, output_folder=None):
        """Instantiation of the NeuralNetwork class.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        params_dict : dict
            Dictionary containing the overall parameters.

        Attributes
        ----------
        self.nn : NeuralNetworkCreator object
            Neural Network architecture creator class for the NN.
        self.data_loader : torch.utils.data.dataloader.DataLoader
            DataLoader for NN training.
        self.outputs : dict
            Dictionary containing the training outputs.
        self.data : numpy.ndarray
            A 2D array containing the stacked snapshots. Eventually converted to
            torch.tensor.
        self.params_dict : dict
            A dictionary containing all the modeling parameters.
        """
        self.created_nn = NeuralNetworkCreator(input_data, target_data, params_dict)
        self.data_loader = None
        self.data_loader_training = None
        self.normalization_technique_class = None
        self.outputs = {}
        self.data = self.created_nn.input_data
        self.nn_params_dict = params_dict["neural_network"]
        if output_folder:
            nn_output_folder = output_folder / Path("neural_network")
            if not os.path.exists(nn_output_folder):
                os.mkdir(nn_output_folder)
            self.output_folder = nn_output_folder

    def __log_run(self, log_type="training"):
        if log_type == "training":
            logger.info(
                "-------------------- Starting training for NeuralNetwork --------------------"
            )
            logger.info(f" # of data: {self.data.shape}")
            logger.info(f" # of layers: {self.nn_params_dict['number_of_hidden_layers']}")
            logger.info(f" NN architecture: {self.created_nn.nn}")

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
            batch_size=self.nn_params_dict[key_batch_size],
            num_workers=self.nn_params_dict[key_num_workers],
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

    @timing_decorator
    def fit(self):
        """Trains Neural Network"""

        self.__log_run("training")
        self.created_nn.apply(self.__init_weights)
        self.data_loader = self.__load_data(
            self.data, "batch_size", "num_workers", True
        )
        loss_function = check_parameters_and_extract(self.nn_params_dict, "loss_function")
        optimizer = check_parameters_and_extract(self.nn_params_dict, "optimizer", extra_param=self.created_nn.parameters())

        epochs = self.nn_params_dict["num_epochs"]

        self.outputs["outputs"] = []
        self.outputs["avg_loss_by_epoch"] = []

        for _ in tqdm(range(epochs)):
            losses_batches_per_epoch = []
            for entry in self.data_loader:
                optimizer.zero_grad()
                reconstructed = self.created_nn(entry)
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
        """Predicts data after trained NeuralNetwork"""
        self.__log_run("predicting")
        with torch.no_grad():
            input_tensor = torch.from_numpy(forward_data.T).float()
            predictions = self.created_nn(input_tensor)
        return predictions.numpy().T

    def plot_quantities_per_epoch(self, quantity, fold):
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
