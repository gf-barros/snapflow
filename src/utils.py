"""Utils modules and data structures for ROM code"""

import torch.nn
import os
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
import yaml
import logging
import time

# Disables log messages when using matplotlib
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.ticker").disabled = True

import logging

# Delete log file and create a logger
try:
    os.remove("log_file.txt")
except OSError:
    pass
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# Create a console handler and set the level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)

# Create a file handler and set the level to DEBUG
file_handler = logging.FileHandler("log_file.txt")
file_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


map_input_function_pytorch = {
    "linear": torch.nn.Linear,
    "leaky_relu": torch.nn.LeakyReLU,
    "sigmoid": torch.nn.Sigmoid,
    "": lambda x: x,
    "smooth_l1_loss": torch.nn.SmoothL1Loss,
    "mse_loss": torch.nn.MSELoss,
    "adam": torch.optim.Adam,
    "kaiming_normal": torch.nn.init.kaiming_normal_,
    "data_loader": DataLoader
}

def check_parameters_and_extract(dict, key, extra_param=None):
    key_params = key + "_parameters"
    if extra_param is None:
        if dict[key_params]["active"]:
            return map_input_function_pytorch[
                dict[key]
            ](**dict[key_params]["parameters"]) 
        else:
            return map_input_function_pytorch[
                dict[key]
            ]() 
    else:
        if dict[key_params]["active"]:
            return map_input_function_pytorch[
                dict[key]
            ](extra_param, **dict[key_params]["parameters"]) 
        else:
            return map_input_function_pytorch[
                dict[key]
            ](extra_param)   
        
def check_parameters_and_extract_layers(dict, key, layer=0):
    key_params = key + "_parameters"
    print(key, dict[key][layer], layer, dict[key_params]["parameters"][layer], map_input_function_pytorch[dict[key][layer]])
    if dict[key_params]["active"][layer]:
        return map_input_function_pytorch[
            dict[key][layer]
        ](**dict[key_params]["parameters"][layer]) 
    else:
        return map_input_function_pytorch[
            dict[key][layer]
        ]() 


def setup_output_folder(params):
    experiment_name = params["experiment_name"]
    experiment_folder = os.path.join("data/output", experiment_name)
    if os.path.exists(experiment_folder):
        shutil.rmtree(experiment_folder)
    output_folder = Path(experiment_folder)
    os.mkdir(output_folder)

    parameters_file_export = output_folder / Path("params.yaml")
    with open(parameters_file_export, "w") as output_files_params:
        yaml.dump(params, output_files_params, sort_keys=False)
    return output_folder


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        if "function_times" not in wrapper.__dict__:
            wrapper.__dict__["function_times"] = {}

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        function_name = func.__name__
        wrapper.__dict__["function_times"][function_name] = elapsed_time

        logger.info(f"{function_name} took {elapsed_time} seconds to execute.")

        return result

    return wrapper
