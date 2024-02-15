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

# ## Tutorial 01 - Autoencoder validation
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
def pipeline_modes(backtest_flag=True, inference_flag=False):
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

    # First Reduction
    svd_train = SVD(
        train_data, params, output_folder=output_folder, analysis_type="train"
    )
    svd_train.fit()
    svd_train.plot_singular_values()
    save_paraview_visualization(svd_train.u[:, 0], output_folder, "train_mode_0")

    svd_test = SVD(test_data, params, output_folder=output_folder, analysis_type="test")
    svd_test.fit()
    svd_test.plot_singular_values()
    save_paraview_visualization(svd_train.u[:, 0], output_folder, "test_mode_0")

    projected_train_data = svd_train.u.T @ train_data
    projected_test_data = svd_test.u.T @ test_data

    if backtest_flag:
        # train_val split
        model_selection_data = data_splitter.split_data(
            projected_train_data, train_test_flag=False
        )

        # fold artifacts:
        for fold in model_selection_data.keys():
            fold_train_data = model_selection_data[fold]["train"]
            fold_train_indices = model_selection_data[fold]["train_indices"]
            fold_validation_data = model_selection_data[fold]["validation"]
            fold_validation_indices = model_selection_data[fold]["validation_indices"]

            # normalize training and validation data
            normalized_projected_train_data, normalization_projected_train_obj = (
                data_normalization(
                    fold_train_data, params, "auto_encoder", transpose=False
                )
            )

            # fit high dimensional data
            auto_encoder = AutoEncoder(
                normalized_projected_train_data, params, output_folder
            )
            auto_encoder.fit()
            auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch", fold)

            # compute error for training data
            normalized_train_predictions = auto_encoder.predict(
                normalized_projected_train_data
            )
            train_predictions = normalization_projected_train_obj.inverse_transform(
                normalized_train_predictions
            )
            high_dimensional_train_predictions = svd_train.u @ train_predictions
            high_dimensional_fold_train_data = svd_train.u @ fold_train_data
            save_paraview_visualization(
                high_dimensional_train_predictions[:, 0],
                output_folder,
                f"postprocessed_prediction_train_{fold_train_indices[0]}_fold_{fold}",
            )
            save_paraview_visualization(
                high_dimensional_fold_train_data[:, 0],
                output_folder,
                f"postprocessed_train_data_{fold_train_indices[0]}_fold_{fold}",
            )
            compute_errors(
                fold,
                high_dimensional_train_predictions,
                high_dimensional_fold_train_data,
                fold_train_indices,
                output_folder,
                analysis_type="train",
                modeling_type="backtest",
            )

            # compute error for validation data
            if fold_validation_data is not None:
                (
                    normalized_projected_validation_data,
                    normalization_projected_validation_obj,
                ) = data_normalization(
                    fold_validation_data, params, "auto_encoder", transpose=False
                )
                normalized_validation_predictions = auto_encoder.predict(
                    normalized_projected_validation_data
                )
                validation_predictions = (
                    normalization_projected_validation_obj.inverse_transform(
                        normalized_validation_predictions
                    )
                )
                high_dimensional_validation_predictions = (
                    svd_train.u @ validation_predictions
                )
                high_dimensional_fold_validation_data = (
                    svd_train.u @ fold_validation_data
                )
                save_paraview_visualization(
                    high_dimensional_validation_predictions[:, 0],
                    output_folder,
                    f"postprocessed_prediction_validation_{fold_validation_indices[0]}_fold_{fold}",
                )
                save_paraview_visualization(
                    high_dimensional_fold_validation_data[:, 0],
                    output_folder,
                    f"postprocessed_validation_data_{fold_validation_indices[0]}_fold_{fold}",
                )
                compute_errors(
                    fold,
                    high_dimensional_validation_predictions,
                    high_dimensional_fold_validation_data,
                    fold_validation_indices,
                    output_folder,
                    analysis_type="validation",
                    modeling_type="backtest",
                )

    if inference_flag:
        # train for all data
        total_train_indices = folded_data[0]["train_indices"]
        total_test_indices = folded_data[0]["test_indices"]

        # normalize training and data
        total_normalized_spatial_train_modes, u_normalization_total_train_obj = (
            data_normalization(total_train_data, params, "svd", transpose=False)
        )
        total_normalized_spatial_test_modes, u_normalization_total_test_obj = (
            data_normalization(total_test_data, params, "svd", transpose=False)
        )
        print(
            f"normalized total spatial train modes dim: {total_normalized_spatial_train_modes.shape}"
        )

        # fit high dimensional data
        auto_encoder = AutoEncoder(
            total_normalized_spatial_train_modes, params, output_folder
        )
        auto_encoder.fit()
        auto_encoder.plot_quantities_per_epoch("avg_loss_by_epoch")

        # compute error for training data
        total_normalized_train_predictions = auto_encoder.predict(
            total_normalized_spatial_train_modes
        )
        total_train_predictions = u_normalization_total_train_obj.inverse_transform(
            total_normalized_train_predictions
        )
        compute_errors(
            fold,
            total_train_predictions,
            0,
            total_train_indices,
            output_folder,
            analysis_type="train",
            modeling_type="inference",
        )

        # compute error for test data
        total_normalized_test_predictions = auto_encoder.predict(
            total_normalized_spatial_test_modes
        )
        total_test_predictions = u_normalization_total_test_obj.inverse_transform(
            total_normalized_test_predictions
        )
        compute_errors(
            fold,
            total_test_predictions,
            0,
            total_test_indices,
            output_folder,
            paraview_plot="first",
            analysis_type="test",
            modeling_type="inference",
        )


# %%capture
pipeline_modes(inference_flag=False)

# +
# TODO: jogar no google docs
# TODO: Save e load do modelo
# TODO: Plotters e cálculos de erros devem ser classe?
# TODO: plots de erro estão errados
# TODO: surrogate
# -
