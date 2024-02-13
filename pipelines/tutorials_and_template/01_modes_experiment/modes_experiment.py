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

# ## Tutorial 01 - Modes Reconstruction
#
# In this tutorial, we are going to test our implementations of the SVD factorization and the AutoEncoder.
#
# We start by running our mandatory preamble for properly defining directions to our source code.

# +
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys

current_working_directory = Path.cwd()
root_directory = current_working_directory.parent.parent.parent
sys.path.append(str(root_directory))
# -

# ### Loading the modules for the pipeline
#
# Then, we load all modules required for this experiment

from yaml import safe_load, YAMLError
from snapflow.utils import setup_output_folder, timing_decorator
from snapflow.snapshots import snapshots_assembly, data_normalization
from snapflow.linear_reduction import SVD
from snapflow.nonlinear_reduction import AutoEncoder
from snapflow.data_split import DataSplitter
from snapflow.postprocessing import compute_errors, save_paraview_visualization

# ### Load parameters file, define experiment name (if needed) and load files

# +
with open("parameters.yaml", "r") as stream:
    try:
        params = safe_load(stream)
    except YAMLError as exc:
        print(exc)

if params["origin_experiment_name"] == "input":
     params["experiment_name"] = input("Experiment name: ")

filenames, snapshots = snapshots_assembly(params["snapshots"])


# -

# ### Setup the pipeline
#
# Now, we create a function to execute the whole pipeline. The function itself is not mandatory, but it could be interesting to isolate the variables into the scope of the pipeline. We are going to time the execution of the pipeline using the `@timing_decorator`.
#
# In this pipeline, we setup two different approaches: the `backtest` and the `inference`. The backtest is often used when we want to evaluate the components of the pipeline separately. It might be necessary to change the ML model, perform hyperparameter tuning, validate hypothesis and so on. Once everything is setup, we can perform the inference, by training our selected model with the training data and testing with the test set. It is good practice to remove the `test` set during backtest, to avoid overfitting into the test set, leading to misleading performance of the model before deploying it into production. 
#
# That is, whenever we are performing backtest, the `train` and `test` are already split at first. Then, during backtest, we split the total `train` data into `train` and `validation` data and perform our backtest. As soon as the model is chosen and defined, we can train the model with the total `train` data and analyze its performance with the `test` set.

@timing_decorator
def pipeline_modes(backtest_flag = True, inference_flag = False):
    # setup directories
    output_folder = setup_output_folder(params)

    # high dimensional data
    svd = SVD(snapshots, params, output_folder)
    svd.fit()
    svd.plot_singular_values()
    save_paraview_visualization(svd.u[:, 0], output_folder, "original_mode_0")

    spatial_modes = svd.u
    print(f"spatial modes dim: {spatial_modes.shape}")

    # train_test split
    data_splitter = DataSplitter(params)
    folded_data = data_splitter.split_data(spatial_modes, train_test_flag=True)
    total_train_data = folded_data[0]["train"]
    total_test_data = folded_data[0]["test"] 
    save_paraview_visualization(total_train_data[:, 0], output_folder, "train_test_split_mode_0")
    print("train_data shape", total_train_data.shape)
    print("train_data type", type(total_train_data))

    if backtest_flag:
        # train_val split
        model_selection_data = data_splitter.split_data(total_train_data, train_test_flag=False)

        # fold artifacts:
        for fold in model_selection_data.keys():
            fold_train_data = model_selection_data[fold]["train"]
            fold_train_indices = model_selection_data[fold]["train_indices"]
            fold_validation_data = model_selection_data[fold]["validation"]
            fold_validation_indices = model_selection_data[fold]["validation_indices"]
            save_paraview_visualization(fold_train_data[:, 0], output_folder, "train_val_split_mode_0")

            # preprocess high dimensional data
            normalized_spatial_train_modes, u_normalization_train_fold_obj = data_normalization(
            fold_train_data, params, "svd", transpose=False
            )    
            save_paraview_visualization(normalized_spatial_train_modes[:, 0], output_folder, "preprocessed_train_split_mode_0")

            # fit high dimensional data
            auto_encoder = AutoEncoder(normalized_spatial_train_modes, params, output_folder)
            auto_encoder.fit()
            auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch", fold)

            # compute error for training data
            normalized_train_predictions = auto_encoder.predict(normalized_spatial_train_modes)
            train_predictions = u_normalization_train_fold_obj.inverse_transform(normalized_train_predictions)
            save_paraview_visualization(train_predictions[:, 0], output_folder, "postprocessed_prediction_split_mode_0")
            compute_errors(fold, train_predictions, fold_train_data, fold_train_indices, output_folder, analysis_type="train", modeling_type="backtest")

            # compute error for validation data
            if fold_validation_data is not None:
                normalized_spatial_val_modes, u_normalization_val_fold_obj = data_normalization(
                fold_validation_data, params, "svd", transpose=False
                )    
                print(f"normalized spatial train modes dim: {normalized_spatial_train_modes.shape}")
                print(f"normalized spatial val modes dim: {normalized_spatial_val_modes.shape}")
                normalized_val_predictions = auto_encoder.predict(normalized_spatial_val_modes)
                val_predictions = u_normalization_val_fold_obj.inverse_transform(normalized_val_predictions)
                compute_errors(fold, val_predictions, fold_validation_data, fold_validation_indices, output_folder, analysis_type="validation", modeling_type="backtest")
                

    if inference_flag:
            # train for all data
            total_train_indices = folded_data[0]["train_indices"]
            total_test_indices = folded_data[0]["test_indices"]

            # normalize training and data
            total_normalized_spatial_train_modes, u_normalization_total_train_obj = data_normalization(
            total_train_data, params, "svd", transpose=False
            )    
            total_normalized_spatial_test_modes, u_normalization_total_test_obj = data_normalization(
            total_test_data, params, "svd", transpose=False
            )    
            print(f"normalized total spatial train modes dim: {total_normalized_spatial_train_modes.shape}")

            # fit high dimensional data
            auto_encoder = AutoEncoder(total_normalized_spatial_train_modes, params, output_folder)
            auto_encoder.fit()
            auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch")

            # compute error for training data
            total_normalized_train_predictions = auto_encoder.predict(total_normalized_spatial_train_modes)
            total_train_predictions = u_normalization_total_train_obj.inverse_transform(total_normalized_train_predictions)
            compute_errors(fold, total_train_predictions, 0, total_train_indices, output_folder, analysis_type="train", modeling_type="inference")

            # compute error for test data
            total_normalized_test_predictions = auto_encoder.predict(total_normalized_spatial_test_modes)
            total_test_predictions = u_normalization_total_test_obj.inverse_transform(total_normalized_test_predictions)
            compute_errors(fold, total_test_predictions, 0, total_test_indices, output_folder, paraview_plot="first", analysis_type="test", modeling_type="inference")

# %%capture
pipeline_modes(inference_flag=False)

# +
# TODO: jogar no google docs
# TODO: Save e load do modelo
# TODO: Plotters e cálculos de erros devem ser classe?
# TODO: plots de erro estão errados
# TODO: surrogate
# -


