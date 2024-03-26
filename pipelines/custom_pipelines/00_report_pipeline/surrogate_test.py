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

# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys

current_working_directory = Path.cwd()
root_directory = current_working_directory.parent.parent.parent
sys.path.append(str(root_directory))

# +
import numpy as np
from yaml import safe_load, YAMLError
from snapflow.utils import setup_output_folder, timing_decorator
from snapflow.snapshots import snapshots_assembly, snapshots_assembly_multiple_folders
from snapflow.normalization import data_normalization
from snapflow.linear_reduction import SVD
from snapflow.nonlinear_reduction import AutoEncoder
from snapflow.surrogate import NeuralNetwork
from snapflow.data_split import DataSplitter
from snapflow.postprocessing import PostProcessing, save_paraview_visualization
from tqdm import tqdm
import os

output_dir = os.environ.get('CODING_DATA_DIR')
print(output_dir)
# -

#/Users/gabriel/Documents/dev/data/snapflow/00_report_pipeline
output_dir += "snapflow/00_report_pipeline/"
output_dir

with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

if params["origin_experiment_name"] == "input":
     params["experiment_name"] = input("Experiment name: ")

angles_to_be_trained = [0, 2, 4, 6, 8]
train_folders_list = ["theta" + str(theta) for theta in angles_to_be_trained]
angles_to_be_tested = [10]
test_folders_list = ["theta" + str(theta) for theta in angles_to_be_tested]

filenames, train_snapshots = snapshots_assembly_multiple_folders(params["snapshots"], 
                                                                 stride=5, 
                                                                 folders_list=train_folders_list, 
                                                                 local=params["local"],
                                                                 export_prefix=output_dir)
print("train snapshots size", train_snapshots.shape)
number_of_train_time_steps = train_snapshots.shape[1]

filenames, test_snapshots = snapshots_assembly_multiple_folders(params["snapshots"], 
                                                                stride=5, 
                                                                folders_list=test_folders_list, 
                                                                local=params["local"],
                                                                export_prefix=output_dir)
print("test snapshots size", test_snapshots.shape)
number_of_test_time_steps = test_snapshots.shape[1]
output_folder = setup_output_folder(params, 
                                    local=params["local"],
                                    export_prefix=output_dir)

# train_test split
data_splitter = DataSplitter(params)
folded_data = data_splitter.split_data(train_snapshots, simple_split=True)
train_data = folded_data[0]["data"]
train_indices = folded_data[0]["indices"]
train_spatial_indices = folded_data[0]["spatial_indices"]

# +
folded_data = data_splitter.split_data(test_snapshots, simple_split=True)
test_data = folded_data[0]["data"]
test_indices = folded_data[0]["indices"]
test_spatial_indices = folded_data[0]["spatial_indices"]

print("train snapshots size", train_data.shape)
print("train indices size", train_indices.shape)
print("test snapshots size", test_data.shape)
print("test indices size", test_indices.shape)

save_paraview_visualization(train_data[:, 0], 
                            output_folder, 
                            "train_split_ic", 
                            local=params["local"],
                            export_prefix=output_dir)
save_paraview_visualization(test_data[:, 0], 
                            output_folder, 
                            "test_split_ic", 
                            local=params["local"],
                            export_prefix=output_dir)

# +
# First Reduction
svd_train = SVD(train_data, params, output_folder=output_folder, analysis_type="train")
svd_train.fit()
svd_train.plot_singular_values()
save_paraview_visualization(svd_train.u[:, 0], 
                            output_folder, 
                            "train_mode_0", 
                            local=params["local"],
                            export_prefix=output_dir)

projected_train_data = svd_train.u.T @ train_data
# -

# normalize training and test data
normalized_projected_train_data, normalization_projected_train_obj = data_normalization(
projected_train_data, params, "auto_encoder", transpose=True
)    
print(f"normalized total projected train data dim: {normalized_projected_train_data.shape}")

# fit high dimensional data
auto_encoder = AutoEncoder(normalized_projected_train_data, params, output_folder)
auto_encoder.fit()
auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch")
auto_encoder.save_model(local=params["local"],
                        export_prefix=output_dir)
latent_train_data = auto_encoder.encode(normalized_projected_train_data)


# +
# latent data transpose
latent_train_data = latent_train_data.T

# normalize training and test data
normalized_latent_train_data, normalization_latent_train_obj = data_normalization(
latent_train_data, params, "surrogate", transpose=True
)    
print(f"normalized total latent train data dim: {normalized_latent_train_data.shape}")

# +
# train surrogate
nn_train_data = np.array([(angle, int(step)) for angle in angles_to_be_trained for step in range(number_of_train_time_steps//len(angles_to_be_trained))])
nn_test_data = np.array([(angle, int(step)) for angle in angles_to_be_tested for step in range(number_of_test_time_steps//len(angles_to_be_tested))])

print("nn_train_data shape", nn_train_data.shape)
print("nn_train_data shape", normalized_latent_train_data.shape)
neural_network = NeuralNetwork(nn_train_data, normalized_latent_train_data, params, output_folder)
neural_network.fit()
neural_network.save_model()
neural_network.plot_quantities_per_epoch("avg_loss_by_epoch", fold=0)

# +
# compute error for training data
normalized_latent_train_predictions = neural_network.predict(nn_train_data)
print("normalized_latent_train_predictions shape", normalized_latent_train_predictions.shape)

latent_train_predictions = normalization_latent_train_obj.inverse_transform(normalized_latent_train_predictions.T)
print("latent_train_predictions shape", latent_train_predictions.shape)

normalized_decoded_train_predictions = auto_encoder.decode(latent_train_predictions.T)
print("normalized_decoded_train_predictions shape", normalized_decoded_train_predictions.shape)

decoded_train_predictions = normalization_projected_train_obj.inverse_transform(normalized_decoded_train_predictions.T)
print("decoded_train_predictions shape", decoded_train_predictions.shape)

train_predictions = svd_train.u @ decoded_train_predictions.T
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
postprocessing_train.compute_errors(local=params["local"],
                        export_prefix=output_dir)

del train_predictions

# +
# compute error for test data
normalized_latent_test_predictions = neural_network.predict(nn_test_data)
latent_test_predictions = normalization_latent_train_obj.inverse_transform(normalized_latent_test_predictions.T)
normalized_decoded_test_predictions = auto_encoder.decode(latent_test_predictions.T)
decoded_test_predictions = normalization_projected_train_obj.inverse_transform(normalized_decoded_test_predictions.T)
test_predictions = svd_train.u @ decoded_test_predictions.T

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

postprocessing_test.compute_errors(local=params["local"],
                        export_prefix=output_dir)
