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

from yaml import safe_load, YAMLError
from src.utils import setup_output_folder, timing_decorator
from src.snapshots import snapshots_assembly, data_normalization
from src.linear_reduction import SVD
from src.nonlinear_reduction import AutoEncoder
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
    

# -

# %%capture
pipeline_modes(inference_flag=False)

# +
# TODO: jogar no google docs
# TODO: Save e load do modelo
# TODO: Plotters e cálculos de erros devem ser classe?
# TODO: plots de erro estão errados
# TODO: surrogate
# -


