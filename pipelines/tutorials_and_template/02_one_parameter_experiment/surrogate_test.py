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

import numpy as np
from yaml import safe_load, YAMLError
from snapflow.utils import setup_output_folder, timing_decorator
from snapflow.snapshots import snapshots_assembly, data_normalization
from snapflow.linear_reduction import SVD
from snapflow.nonlinear_reduction import AutoEncoder
from snapflow.surrogate import NeuralNetwork
from snapflow.data_split import DataSplitter
from snapflow.postprocessing import PostProcessing, save_paraview_visualization
from tqdm import tqdm

with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

if params["origin_experiment_name"] == "input":
     params["experiment_name"] = input("Experiment name: ")


snapshots_params = params["snapshots"]
snapshots_dir = Path(snapshots_params["folder"])
filename = snapshots_dir / Path(snapshots_params["file_name_contains"][0] + f"_{str(0)}.npy")
train_snapshots = np.load(filename)
train_snapshots = train_snapshots[:, ::5]
number_of_train_time_steps = train_snapshots.shape[1]


print("train snapshots size", train_snapshots.shape)

for i in tqdm([2, 4, 6, 8]):
    filename = snapshots_dir / Path(snapshots_params["file_name_contains"][0] + f"_{str(i)}.npy")
    snapshots_temp = np.load(filename)
    snapshots_temp = snapshots_temp[:, ::5]
    train_snapshots = np.hstack((train_snapshots, snapshots_temp))
del snapshots_temp

print("train snapshots size", train_snapshots.shape)

filename = snapshots_dir / Path(snapshots_params["file_name_contains"][0] + f"_{str(10)}.npy")
test_snapshots = np.load(filename)
test_snapshots = test_snapshots[:, ::5]
number_of_test_time_steps = test_snapshots.shape[1]

print("test snapshots size", test_snapshots.shape)
# +
@timing_decorator
def pipeline_surrogate():
    # setup directories
    output_folder = setup_output_folder(params)

    # train_test split
    data_splitter = DataSplitter(params)
    folded_data = data_splitter.split_data(train_snapshots, simple_split=True)
    train_data = folded_data[0]["data"]
    train_indices = folded_data[0]["indices"]
    train_spatial_indices = folded_data[0]["spatial_indices"]

    folded_data = data_splitter.split_data(test_snapshots, simple_split=True)
    test_data = folded_data[0]["data"]
    test_indices = folded_data[0]["indices"]
    test_spatial_indices = folded_data[0]["spatial_indices"]

    print("train snapshots size", train_data.shape)
    print("train indices size", train_indices.shape)
    print("test snapshots size", test_data.shape)
    print("test indices size", test_indices.shape)

    save_paraview_visualization(train_data[:, 0], output_folder, "train_split_ic")
    save_paraview_visualization(test_data[:, 0], output_folder, "test_split_ic")


    print("First SVD")
    # First Reduction
    svd_train = SVD(train_data, params, output_folder=output_folder, analysis_type="train")
    svd_train.fit()
    svd_train.plot_singular_values()
    save_paraview_visualization(svd_train.u[:, 0], output_folder, "train_mode_0")

    print("Second SVD")
    svd_test = SVD(test_data, params, output_folder=output_folder, analysis_type="test")
    svd_test.fit()
    svd_test.plot_singular_values()
    save_paraview_visualization(svd_test.u[:, 0], output_folder, "test_mode_0")

    projected_train_data = svd_train.u.T @ train_data
    projected_test_data = svd_test.u.T @ test_data

    # normalize training and test data
    normalized_projected_train_data, normalization_projected_train_obj = data_normalization(
    projected_train_data, params, "auto_encoder", transpose=False
    )    
    normalized_projected_test_data, normalization_projected_test_obj = data_normalization(
    projected_test_data, params, "auto_encoder", transpose=False
    )    
    print(f"normalized total projected train data dim: {normalized_projected_train_data.shape}")

    # fit high dimensional data
    auto_encoder = AutoEncoder(normalized_projected_train_data, params, output_folder)
    auto_encoder.fit()
    auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch")
    latent_train_data = auto_encoder.encode(normalized_projected_train_data)
    latent_test_data = auto_encoder.encode(normalized_projected_test_data)

    # latent data transpose
    latent_train_data = latent_train_data.T
    latent_test_data = latent_test_data.T

    # normalize training and test data
    normalized_latent_train_data, normalization_latent_train_obj = data_normalization(
    latent_train_data, params, "surrogate", transpose=False
    )    
    normalized_latent_test_data, normalization_latent_test_obj = data_normalization(
    latent_test_data, params, "surrogate", transpose=False
    )    
    print(f"normalized total latent train data dim: {normalized_latent_train_data.shape}")

    # train surrogate
    nn_train_data = np.array([(angle, int(step)) for angle in [0, 2, 4, 6, 8] for step in range(number_of_train_time_steps)])
    nn_test_data = np.array([(angle, int(step)) for angle in [10] for step in range(number_of_test_time_steps)])

    print("nn_train_data shape", nn_train_data.shape)
    neural_network = NeuralNetwork(nn_train_data, normalized_latent_train_data, params, output_folder)
    neural_network.fit()
    neural_network.plot_quantities_per_epoch("avg_loss_by_epoch", fold=0)

    # compute error for training data
    normalized_latent_train_predictions = neural_network.predict(nn_train_data)
    print("normalized_latent_train_predictions shape", normalized_latent_train_predictions.shape)

    latent_train_predictions = normalization_latent_train_obj.inverse_transform(normalized_latent_train_predictions)
    print("latent_train_predictions shape", latent_train_predictions.shape)

    normalized_decoded_train_predictions = auto_encoder.decode(latent_train_predictions)
    print("normalized_decoded_train_predictions shape", normalized_decoded_train_predictions.shape)

    decoded_train_predictions = normalization_projected_train_obj.inverse_transform(normalized_decoded_train_predictions)
    print("decoded_train_predictions shape", decoded_train_predictions.shape)

    train_predictions = svd_train.u @ decoded_train_predictions
    print("decoded_train_predictions shape", decoded_train_predictions.shape)

    postprocessing_train = PostProcessing(fold=0, 
                    predictions=train_predictions, 
                    ground_truth=train_data, 
                    indices=train_indices, 
                    spatial_indices=train_spatial_indices,
                    output_folder=output_folder, 
                    params_dict=params,
                    analysis_type="train", 
                    modeling_type="inference"
                    )
    postprocessing_train.compute_errors()

    del train_predictions

    # compute error for test data
    normalized_latent_test_predictions = neural_network.predict(nn_test_data)
    latent_test_predictions = normalization_latent_test_obj.inverse_transform(normalized_latent_test_predictions)
    normalized_decoded_test_predictions = auto_encoder.decode(latent_test_predictions)
    decoded_test_predictions = normalization_projected_test_obj.inverse_transform(normalized_decoded_test_predictions)
    test_predictions = svd_test.u @ decoded_test_predictions

    postprocessing_test = PostProcessing(fold=0, 
                    predictions=test_predictions, 
                    ground_truth=test_data, 
                    indices=test_indices, 
                    spatial_indices=test_spatial_indices,
                    params_dict=params,
                    output_folder=output_folder, 
                    analysis_type="test", 
                    modeling_type="inference"
                    )

    postprocessing_test.compute_errors()

# %%capture
pipeline_surrogate()