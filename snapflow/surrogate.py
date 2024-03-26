import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from snapflow.utils import map_input_function_pytorch, check_parameters_and_extract
from snapflow.utils import logger, timing_decorator
from pathlib import Path
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data.T, dtype=torch.float32)  
        self.dataset = TensorDataset(self.input_data, self.target_data)
        logger.info(self.input_data.shape)      
        logger.info(self.target_data.shape)      
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
    
    def __create_nn_layer(self, layer_input_size, layer_output_size, layer_number=0, layer_type="hidden"):
        self.layers_nn.append(
            map_input_function_pytorch["linear"](
                layer_input_size,
                layer_output_size,
            )
        )
        if self.nn_params_dict[f"{layer_type}_layers_activation_function_parameters"]["active"][layer_number]: 
            self.layers_nn.append(
                map_input_function_pytorch[
                    self.nn_params_dict[f"{layer_type}_layers_activation_function"][layer_number]
                ](**self.nn_params_dict[f"{layer_type}_layers_activation_function_parameters"]["parameters"][layer_number])
            )
        else: 
            self.layers_nn.append(
                map_input_function_pytorch[
                    self.nn_params_dict[f"{layer_type}_layers_activation_function"][layer_number]
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
        
        self.__create_nn_layer(self.nn_params_dict["hidden_layers_sizes"][layer_number], 
                                target_size,
                                0,
                                "output")

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
        self.dataset = self.created_nn.dataset
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
        self.data_loader = check_parameters_and_extract(self.nn_params_dict, "data_loader", extra_param=self.dataset)
        loss_function = check_parameters_and_extract(self.nn_params_dict, "loss_function")
        optimizer = check_parameters_and_extract(self.nn_params_dict, "optimizer", extra_param=self.created_nn.parameters())

        epochs = self.nn_params_dict["num_epochs"]

        self.outputs["outputs"] = []
        self.outputs["avg_loss_by_epoch"] = []

        logger.info(self.data_loader)

        for _ in tqdm(range(epochs)):
            losses_batches_per_epoch = []
            for entry, target in self.data_loader:
                optimizer.zero_grad()
                predictions = self.created_nn(entry)
                loss = loss_function(predictions, target)
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
            input_tensor = torch.from_numpy(forward_data).float()
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

    def save_model(self, 
                local=True,
                export_prefix=None):
        """
        Saves the model's state dictionary to the specified file path.
        :param file_path: The path to save the model to.
        """
        if not local:
            filepath = Path(export_prefix) / Path(self.nn_params_dict["folder"]) / Path("nn.pth")
        else:
            filepath = Path(self.nn_params_dict["folder"]) / Path("nn.pth")
        torch.save(self.created_nn.state_dict(), filepath)
        logger.info(f'Model saved to {filepath}')

    def load_model(self):
            """
            Loads the model's state dictionary from the specified file path.
            :param file_path: The path to load the model from.
            """
            filepath = Path(self.nn_params_dict["folder"]) / Path("nn.pth")
            self.created_nn.load_state_dict(torch.load(filepath))
            self.eval()  # Set the model to evaluation mode
            logger.info(f'Model loaded from {filepath}')



class DMD:
    """
        Instantiation of the DMD class. This class inherits from the SVD 
        class and utilizes its attributes such as 'U', 'S', and 'V' for 
        DMD-specific computations.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        params : dict
            Dictionary containing the overall parameters.
        output_folder : Path
            Path containing the output folder

        Attributes
        ----------
        self.svd.u : numpy.ndarray
            A 2D array containing the left singular vectors modes from self.data.
        self.svd.s : numpy.ndarray
            A 1D array containing the singular values from self.data.
        self.svd.vt : numpy.ndarray
            A 2D array containing the right singular vectors modes from self.data.
        self.snapshots_matrix : numpy.ndarray
            A 2D array containing the stacked snapshots.
        self.dmd_params_dict : dict
            A dictionary containing all the modeling parameters.
        self.snapshots_matrix : ...
            ....
            ....

        Returns
        -------
        self.dmd_approximation : Dict
            Dictionary containing modes, eigenvalues, singular values and approximate solution.

    """
    def __init__(self, svd, data, target, params, output_folder):
        self.svd = svd
        self.dmd_params_dict = params["dmd"]
        self.snapshots_x1 = data
        self.snapshots_x2 = target
        self.dmd_approximation = {}
        if output_folder:
            dmd_output_folder = output_folder / Path("dmd")
            if not os.path.exists(dmd_output_folder):
                os.mkdir(dmd_output_folder)
            self.output_folder = dmd_output_folder

    @timing_decorator
    def fit(self):
        self.svd.s = np.divide(1.0, self.svd.s)
        self.svd.s = np.diag(self.svd.s)
        self.svd.u = np.transpose(self.svd.u)
        self.svd.vt = np.transpose(self.svd.vt)
        a_tilde = np.linalg.multi_dot(
            [self.svd.u, self.snapshots_x2, self.svd.vt, self.svd.s]
        )
        self.dmd_approximation["eigenvals_original"], eigenvec = np.linalg.eig(a_tilde)
        eigenval = np.log(self.dmd_approximation["eigenvals_original"]) / (
            self.dmd_params_dict["dt"]
        )
        self.dmd_approximation["eigenvals_processed"] = eigenval
        phi_dmd = np.linalg.multi_dot(
            [self.snapshots_x2, self.svd.vt, self.svd.s, eigenvec]
        )
        phi_inv = np.linalg.pinv(phi_dmd)
        initial_vector = self.snapshots_x1[:, 0]
        b_vector = np.dot(phi_inv, initial_vector)
        b_vector = b_vector[:, np.newaxis]
        t_vector = (
            np.arange(start=self.dmd_params_dict["dmd_start"], stop=self.dmd_params_dict["dmd_end"])
            * self.dmd_params_dict["dt"]
        )
        t_vector = t_vector[np.newaxis, :]
        self.dmd_approximation["t"] = t_vector
        self.dmd_approximation["eigenvals_processed"] = eigenval
        eigenval = eigenval[:, np.newaxis]
        temp = np.multiply(eigenval, t_vector)
        temp = np.exp(temp)
        dynamics = np.multiply(b_vector, temp)
        x_dmd = np.dot(phi_dmd, dynamics)
        self.dmd_approximation["dmd_matrix"] = x_dmd

    def plot_eigenvalues(self):
        data = self.dmd_approximation["eigenvals_original"]
        real_part = np.real(data)
        imag_part = np.imag(data)

        # Plotting scatter plot using matplotlib
        fig, ax = plt.subplots()

        # Plotting the unitary circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", color="grey")

        # Plotting the real and imaginary parts of eigenvalues
        ax.scatter(real_part, imag_part)

        # Setting x and y axis labels
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")

        # Setting axis limits to -1.5 and 1.5 to fit the unitary circle
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        plt.gca().set_aspect("equal", adjustable="box")

        if hasattr(self, "output_folder"):
            plt.savefig(self.output_folder / Path(f"eigenvalues.png"))
        plt.close()
    

