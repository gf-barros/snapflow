"""Utils modules and data structures for ROM code"""

import torch.nn

map_input_function_pytorch = {
    "linear": torch.nn.Linear,
    "leaky_relu": torch.nn.LeakyReLU,
    "sigmoid": torch.nn.Sigmoid,
    "": lambda x: x,
    "smooth_l1_loss": torch.nn.SmoothL1Loss,
}
