# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: turbiditos_surrogate_vscode
#     language: python
#     name: turbiditos_surrogate
# ---

# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys

current_working_directory = Path.cwd()
root_directory = current_working_directory.parent.parent.parent
sys.path.append(str(root_directory))

from yaml import safe_load, YAMLError
import numpy as np
from snapflow.utils import setup_output_folder, timing_decorator
from snapflow.snapshots import snapshots_assembly_multiple_folders
from snapflow.linear_reduction import SVD
from snapflow.data_split import DataSplitter
from snapflow.postprocessing import PostProcessing, save_paraview_visualization
from pydmd import DMD
from pydmd.preprocessing import zero_mean_preprocessing
from pydmd.plotter import plot_summary
from tqdm import tqdm
import os

with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

if params["origin_experiment_name"] == "input":
    params["experiment_name"] = input("Experiment name: ")

output_dir = os.environ.get('CODING_DATA_DIR')
print(output_dir)
# -

#/Users/gabriel/Documents/dev/data/snapflow/00_report_pipeline
output_dir += "snapflow/"
train_folders_list = ["theta0"]


filenames, snapshots = snapshots_assembly_multiple_folders(params["snapshots"], 
                                                                 stride=1, 
                                                                 folders_list=train_folders_list, 
                                                                 local=params["local"],
                                                                 export_prefix=output_dir)
snapshots_x1 = snapshots[:, :-1]
snapshots_x2 = snapshots[:, 1:]

# +
@timing_decorator
def pipeline_dmd():
    # setup directories
    output_folder = setup_output_folder(params)

    # train_test split
    data_splitter = DataSplitter(params)
    folded_data = data_splitter.split_data(snapshots_x1, simple_split=True)
    train_data = folded_data[0]["data"]
    train_indices = folded_data[0]["indices"]
    train_spatial_indices = folded_data[0]["spatial_indices"]

    print("train snapshots size", train_data.shape)
    print("train indices size", train_indices.shape)

    print(train_data[:, 0].shape)
    print(output_folder)
    save_paraview_visualization(train_data[:, 0], output_folder, "train_split_ic")

    print("First SVD")
    # First Reduction
    dmd = zero_mean_preprocessing(DMD(svd_rank=250, exact=True))
    dmd.fit(snapshots)
    train_reconstruction = dmd.reconstructed_data.real    
    save_paraview_visualization(train_reconstruction[:, -1], output_folder, "prediction_last_step")
    save_paraview_visualization(snapshots_x1[:, -1], output_folder, "original_last_step")
    print(np.linalg.norm(train_reconstruction - snapshots_x1)/np.linalg.norm(snapshots_x1))
    # -

# %%capture
pipeline_dmd()

# -


