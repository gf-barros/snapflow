"""Utils modules and data structures for ROM code"""

import torch.nn
import os
import shutil
from pathlib import Path
import yaml
from collections import OrderedDict

map_input_function_pytorch = {
    "linear": torch.nn.Linear,
    "leaky_relu": torch.nn.LeakyReLU,
    "sigmoid": torch.nn.Sigmoid,
    "": lambda x: x,
    "smooth_l1_loss": torch.nn.SmoothL1Loss,
}


def setup_output_folder(params):
    experiment_name = params["experiment_name"]
    experiment_folder = os.path.join("data/output", experiment_name)
    if os.path.exists(experiment_folder):
        shutil.rmtree(experiment_folder)
    output_folder = Path(experiment_folder)
    os.mkdir(output_folder)

    parameters_file_export = output_folder / Path('params.yaml')
    with open(parameters_file_export, 'w') as output_files_params:
        yaml.dump(params, output_files_params, sort_keys=False) 
    return output_folder         