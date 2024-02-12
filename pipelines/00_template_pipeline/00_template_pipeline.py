# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: turbiditos_surrogate
#     language: python
#     name: python3
# ---

# ## Pipeline template
#
# Both Jupyter Notebook and Python script should have the same code. Sync is done via Jupytext library.

# ### Preamble
#
# This preamble is required for every pipeline, regardless of being executed via Jupyter Notebook or Python script. It is responsible for adding the root directory into the system PATH during execution. 

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys

current_working_directory = Path.cwd()
root_directory = current_working_directory.parent.parent
sys.path.append(str(root_directory))
# -

# ### Loading the modules
# Modules should be loaded as follows:


from yaml import safe_load, YAMLError
from snapflow.utils import setup_output_folder, timing_decorator
from snapflow.snapshots import snapshots_assembly, data_normalization
from snapflow.linear_reduction import SVD
from snapflow.nonlinear_reduction import AutoEncoder
from snapflow.data_split import DataSplitter
from snapflow.postprocessing import compute_errors, save_paraview_visualization

# ### Reading the YAML file containing the parameters:
#
# Every pipeline should have its own parameters YAML file following the one presented in this template. It should be read using the following block of code:

with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

# ### Experiment name
# Notice that each pipeline can have multiple experiments. Each experiment should have its own name for output dumping purposes. If the `origin_experiment_name` key on the parameters file returns `input` (specially for debugging), the terminal will request a name for that experiment. 

if params["origin_experiment_name"] == "input":
     params["experiment_name"] = input("Experiment name: ")

params

# ### Loading the data
# For snapshots existent in the simulation output files, the `fenics_h5` and `libmesh_h5` files. Also, loading `.npy` is possible. If data is available in any other file type, the pipeline can be used as long as the snapshots are stacked on a $n \times m$ matrix, where `n` the spatial discretization of the vector and `m` is the number of snapshots.

filenames, snapshots = snapshots_assembly(params["snapshots"])

# +
# TODO: jogar no google docs
# TODO: Save e load do modelo
# TODO: Plotters e cálculos de erros devem ser classe?
# TODO: plots de erro estão errados
# TODO: surrogate
# TODO: create_pipeline script -> create folders, gitkeeps, headers, notebooks and scripts with preambles
