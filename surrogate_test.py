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

import numpy as np
from yaml import safe_load, YAMLError
from src.utils import setup_output_folder, timing_decorator
from src.snapshots import snapshots_assembly, data_normalization
from src.linear_reduction import SVD
from src.nonlinear_reduction import AutoEncoder
from src.surrogate import NeuralNetwork
from src.data_split import DataSplitter
from src.postprocessing import compute_errors, save_paraview_visualization

with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

if params["origin_experiment_name"] == "input":
     params["experiment_name"] = input("Experiment name: ")

filenames, snapshots = snapshots_assembly(params["snapshots"])

# +
# @timing_decorator
# def pipeline_surrogate():
# setup directories
output_folder = setup_output_folder(params)

# train_test split
data_splitter = DataSplitter(params)
folded_data = data_splitter.split_data(snapshots, train_test_flag=True)
train_data = folded_data[0]["train"]
test_data = folded_data[0]["test"] 
total_train_indices = folded_data[0]["train_indices"]
total_test_indices = folded_data[0]["test_indices"]
save_paraview_visualization(train_data[:, 0], output_folder, "train_split_ic")
save_paraview_visualization(test_data[:, 0], output_folder, "test_split_ic")

print("train_data shape", train_data.shape)
print("train_data type", type(train_data))

# First Reduction
svd_train = SVD(train_data, params, output_folder=output_folder, analysis_type="train")
svd_train.fit()
svd_train.plot_singular_values()
save_paraview_visualization(svd_train.u[:, 0], output_folder, "train_mode_0")

svd_test = SVD(test_data, params, output_folder=output_folder, analysis_type="test")
svd_test.fit()
svd_test.plot_singular_values()
save_paraview_visualization(svd_train.u[:, 0], output_folder, "test_mode_0")

projected_train_data = svd_train.u.T @ train_data
projected_test_data = svd_test.u.T @ test_data


# normalize training and test data
normalized_projected_train_data, normalization_projected_train_obj = data_normalization(
projected_train_data, params, "auto_encoder", transpose=False
)    
normalized_projected_test_data, normalization_projected_test_obj = data_normalization(
projected_test_data, params, "auto_encoder", transpose=False
)    
print(f"normalized total spatial train modes dim: {normalized_projected_train_data.shape}")

# fit high dimensional data
auto_encoder = AutoEncoder(normalized_projected_train_data, params, output_folder)
auto_encoder.fit()
auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch")
latent_train_data = auto_encoder.encode(normalized_projected_train_data)
latent_test_data = auto_encoder.encode(normalized_projected_test_data)

# normalize training and test data
normalized_latent_train_data, normalization_latent_train_obj = data_normalization(
latent_train_data, params, "surrogate", transpose=False
)    
normalized_latent_test_data, normalization_latent_test_obj = data_normalization(
latent_test_data, params, "surrogate", transpose=False
)    
print(f"normalized total spatial train modes dim: {normalized_projected_train_data.shape}")

# train surrogate
nn_train_data = np.array([(angle, step) for angle in [0] for step in total_train_indices])
nn_test_data = np.array([(angle, step) for angle in [0] for step in total_test_indices])

neural_network = NeuralNetwork(nn_train_data, normalized_latent_train_data, params, output_folder)
neural_network.fit()
neural_network.plot_quantities_per_epoch("avg_loss_by_epoch")

# compute error for training data
normalized_latent_train_predictions = neural_network.predict(normalized_latent_train_data)
latent_train_predictions = normalization_latent_train_obj.inverse_transform(normalized_latent_train_predictions)
normalized_decoded_train_predictions = auto_encoder.decode(latent_train_predictions)
decoded_train_predictions = normalization_projected_train_obj.inverse_transform(normalized_decoded_train_predictions)
train_predictions = svd_train.u @ decoded_train_predictions

compute_errors(fold=0, 
                prediction=train_predictions, 
                ground_truth=latent_train_data, 
                indices=total_train_indices, 
                output_folder=output_folder, 
                analysis_type="train", 
                modeling_type="inference"
                )

# compute error for test data
normalized_latent_test_predictions = neural_network.predict(normalized_latent_test_data)
latent_test_predictions = normalization_latent_test_obj.inverse_transform(normalized_latent_test_predictions)
normalized_decoded_test_predictions = auto_encoder.decode(latent_test_predictions)
decoded_test_predictions = normalization_projected_test_obj.inverse_transform(normalized_decoded_test_predictions)
test_predictions = svd_test.u @ decoded_test_predictions

compute_errors(fold=0, 
                prediction=test_predictions, 
                ground_truth=latent_test_data, 
                indices=total_test_indices, 
                output_folder=output_folder, 
                analysis_type="test", 
                modeling_type="inference"
                )
# -

# %%capture
#pipeline_surrogate()

# +
# TODO: jogar no google docs
# TODO: Save e load do modelo
# TODO: Plotters e cálculos de erros devem ser classe?
# TODO: plots de erro estão errados
# TODO: surrogate
# -


