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
from snapflow.snapshots import snapshots_assembly, data_normalization
from snapflow.linear_reduction import SVD
from snapflow.nonlinear_reduction import AutoEncoder
from snapflow.data_split import DataSplitter
from snapflow.postprocessing import PostProcessing, save_paraview_visualization
from snapflow.surrogate import DMD
from tqdm import tqdm

with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

if params["origin_experiment_name"] == "input":
    params["experiment_name"] = input("Experiment name: ")

filenames, snapshots = snapshots_assembly(params["snapshots"])
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
    svd_train = SVD(train_data, params, output_folder=output_folder, analysis_type="train")
    svd_train.fit()
    svd_train.plot_singular_values()
    save_paraview_visualization(svd_train.u[:, -1], output_folder, "train_mode_0")

    print(svd_train.u.shape)
    print(svd_train.s.shape)
    print(svd_train.vt.shape)


    dmd_train = DMD(svd_train, train_data, snapshots_x2, params, output_folder)
    dmd_train.fit()
    dmd_train.plot_eigenvalues()
    train_predictions = dmd_train.dmd_approximation["dmd_matrix"]

    save_paraview_visualization(np.real(train_predictions[:, -1]), output_folder, "pred_test")
    print(train_spatial_indices)
    postprocessing = PostProcessing(
                    fold=0, 
                    predictions=train_predictions, 
                    ground_truth=train_data, 
                    indices=train_indices, 
                    spatial_indices=train_spatial_indices,
                    output_folder=output_folder, 
                    params_dict=params,
                    analysis_type="test", 
                    modeling_type="inference"
                    )
    postprocessing.compute_errors()
    # -

# %%capture
pipeline_dmd()

# -


