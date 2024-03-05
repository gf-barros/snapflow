import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from pathlib import Path
import shutil
import h5py
from snapflow.utils import logger, timing_decorator


def write_dict_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as csvfile:
        fieldnames = data_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

def create_l2_error_line_chart(l2_norm_dict, fold, output_folder, analysis_type="train", modeling_type="backtest", clip=None):
    max_index = max(l2_norm_dict, key=l2_norm_dict.get)
    max_value = l2_norm_dict[max_index]
    indices = list(l2_norm_dict.keys())
    values = list(l2_norm_dict.values())

    plt.plot(indices, values, label="Values")            
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"L2 norm error for fold {fold}")
    plt.axhline(
        y=max_value, color="red", linestyle="--", label=f"Max Value ({max_value:.2f})"
    )

    if output_folder:
        if not clip: 
            plt.savefig(output_folder / Path(f"l2_error_line_{modeling_type}_{analysis_type}_fold_{fold}_full.png"))
        else:
            plt.savefig(output_folder / Path(f"l2_error_line_{modeling_type}_{analysis_type}_fold_{fold}_clip_{str(clip).replace('.','')}.png"))
    plt.close()

def create_l2_error_bar_chart(l2_norm_dict, fold, output_folder, analysis_type="train", modeling_type="backtest", clip=None):
    max_index = max(l2_norm_dict, key=l2_norm_dict.get)
    max_value = l2_norm_dict[max_index]
    indices = list(l2_norm_dict.keys())
    values = list(l2_norm_dict.values())

    plt.bar(indices, values, label="Values")           
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"L2 norm error for fold {fold}")
    plt.axhline(
        y=max_value, color="red", linestyle="--", label=f"Max Value ({max_value:.2f})"
    )

    if output_folder:
        if not clip: 
            plt.savefig(output_folder / Path(f"l2_error_bar_{modeling_type}_{analysis_type}_fold_{fold}_full.png"))
        else:
            plt.savefig(output_folder / Path(f"l2_error_bar_{modeling_type}_{analysis_type}_fold_{fold}_clip_{str(clip).replace('.','')}.png"))
    plt.close()


def copy_and_paste_paraview_directory(source_path, destination_path):
    try:
        shutil.copy(
            source_path / Path("concentration_1.h5"),
            destination_path / Path("concentration_1.h5"),
        )
        shutil.copy(
            source_path / Path("concentration_1.xdmf"),
            destination_path / Path("concentration_1.xdmf"),
        )
    except:
        pass


def copy_and_paste_log_file_directory(source_path, destination_path):
    try:
        shutil.copy(
            source_path / Path("log_file.txt"), destination_path / Path("log_file.txt")
        )
    except:
        pass


def insert_h5_vector(vector, paraview_output_folder):
    """Injects vector on H5 files for post-processing viz."""
    filename_output = Path(
        os.path.join(
            paraview_output_folder,
            "concentration_1.h5",
        )
    )
    with h5py.File(filename_output, "r+") as h5_file_output:
        h5_file_output["concentration"]["concentration_0"]["vector"][...] = vector[
            :, np.newaxis
        ]


def save_paraview_visualization(vector, output_folder, plot_name):
    paraview_input_folder = Path(os.path.join("data/visualization"))
    if output_folder:
        paraview_output_folder = output_folder / Path("paraview_plots")
        if not os.path.exists(paraview_output_folder):
            os.mkdir(paraview_output_folder)

    paraview_output_export_folder = paraview_output_folder / Path(f"{plot_name}")
    if not os.path.exists(paraview_output_export_folder):
        os.mkdir(paraview_output_export_folder)
    copy_and_paste_paraview_directory(
        paraview_input_folder, paraview_output_export_folder
    )
    insert_h5_vector(vector, paraview_output_export_folder)

def compute_l2_norm_error(ground_truth_df, prediction_df):
    l2_norm_error = {}
    for column in ground_truth_df.columns:
        diff = ground_truth_df[column] - prediction_df[column]
        norm_diff = np.linalg.norm(diff)
        norm_original = np.linalg.norm(ground_truth_df[column])

        # Avoid division by zero
        if norm_original != 0:
            l2_norm_error[column] = norm_diff / norm_original
        else:
            l2_norm_error[column] = np.nan  # or some other placeholder value indicating undefined error

    return l2_norm_error

def compute_clipped_l2_norm_error(ground_truth_df, prediction_df, clip=None):
    l2_norm_error = {}
    for column in ground_truth_df.columns:
        filtered_gt_df = ground_truth_df.loc[ground_truth_df[column] > clip]
        filtered_pred_df = prediction_df.loc[filtered_gt_df.index]

        diff = filtered_gt_df[column] - filtered_pred_df[column]
        norm_diff = np.linalg.norm(diff)
        norm_original = np.linalg.norm(filtered_gt_df[column])

        # Avoid division by zero
        if norm_original != 0:
            l2_norm_error[column] = norm_diff / norm_original
        else:
            l2_norm_error[column] = np.nan  # or some other placeholder value indicating undefined error

    return l2_norm_error

def save_paraview_largest_errors(l2_norm_error_dict, ground_truth_df, prediction_df, spatial_indices, output_folder, fold, analysis_type="train", modeling_type="backtest", clip="none"):
    # Finding the key with the largest error
    key_largest_error = max(l2_norm_error_dict, key=l2_norm_error_dict.get)
    ground_truth_largest_error = ground_truth_df.loc[spatial_indices, key_largest_error].values
    prediction_largest_error = prediction_df.loc[spatial_indices, key_largest_error].values

    # Saving solutions for largest error
    save_paraview_visualization(
                ground_truth_largest_error,
                output_folder,
                f"ground_truth_largest_error_{modeling_type}_{analysis_type}_fold_{fold}_clip_{str(clip).replace('.','')}_{key_largest_error}",
            )
    
    save_paraview_visualization(
                prediction_largest_error,
                output_folder,
                f"prediction_largest_error_{modeling_type}_{analysis_type}_fold_{fold}_clip_{str(clip).replace('.','')}_{key_largest_error}",            )

def save_paraview_smallest_errors(l2_norm_error_dict, ground_truth_df, prediction_df, spatial_indices, output_folder, fold, analysis_type="train", modeling_type="backtest", clip="none"):
    # Finding the key with the smallest error
    key_smallest_error = min(l2_norm_error_dict, key=l2_norm_error_dict.get)
    ground_truth_smallest_error = ground_truth_df.loc[spatial_indices, key_smallest_error].values
    prediction_smallest_error = prediction_df.loc[spatial_indices, key_smallest_error].values

    # Saving solutions for smallest error
    save_paraview_visualization(
                ground_truth_smallest_error,
                output_folder,
                f"ground_truth_smallest_error_{modeling_type}_{analysis_type}_fold_{fold}_clip_{str(clip).replace('.','')}_{key_largest_error}",            )
    
    save_paraview_visualization(
                prediction_smallest_error,
                output_folder,
                f"prediction_smallest_error_{modeling_type}_{analysis_type}_fold_{fold}_clip_{str(clip).replace('.','')}_{key_largest_error}",            )


@timing_decorator
def compute_errors(
    fold,
    prediction,
    ground_truth,
    indices,
    spatial_indices,
    output_folder,
    params,
    analysis_type="train",
    modeling_type="backtest",
):
    logger.info(
        "-------------------- Computing Errors and Metrics --------------------"
    )
    errors_params = params["metrics"]

    output_dict = {}
    if output_folder:
        general_output_folder = output_folder / Path("general_outputs")
        if not os.path.exists(general_output_folder):
            os.mkdir(general_output_folder)

    # Frobenius norm
    if errors_params.get("frobenius_norm"):
        logger.info("-------------------- Computing Frobenius norm--------------------")
        frobenius_norm = np.linalg.norm(prediction - ground_truth) / np.linalg.norm(
            ground_truth
        )
        output_dict[fold] = {}
        output_dict[fold]["analysis_type"] = analysis_type
        output_dict[fold]["frobenius_rel_error"] = frobenius_norm
        dict_path = general_output_folder / Path(
            f"general_{modeling_type}_{analysis_type}.csv"
        )
        write_dict_to_csv(dict_path, output_dict)

    # L2 norm across all data
    if errors_params.get("l2_error_in_sequence").get("active"):
        logger.info(
            "-------------------- Computing L2 Norm across all data --------------------"
        )
        ground_truth_df = pd.DataFrame(ground_truth, columns=indices)
        prediction_df = pd.DataFrame(prediction, columns=indices)

        l2_norm_error_dict = {}

        clips = errors_params.get("l2_error_in_sequence").get("clip")
        if clips is None:
            l2_norm_error_dict[0] = compute_l2_norm_error(ground_truth_df, prediction_df) 
        else:
            for clip in clips:
                l2_norm_error_dict[clip] = compute_clipped_l2_norm_error(ground_truth_df, prediction_df, clip=clip) 

    # Plot L2 bar chart across all data
    if errors_params.get("l2_error_in_sequence").get("bar_plot_error"):
        logger.info(
            "-------------------- Computing L2 error bar chart --------------------"
        )

        for key in l2_norm_error_dict:
            create_l2_error_bar_chart(l2_norm_error_dict[key], fold, output_folder, analysis_type, modeling_type, clip=key)

    # Plot L2 line chart across all data
    if errors_params.get("l2_error_in_sequence").get("line_plot_error"):
        logger.info(
            "-------------------- Computing L2 error line chart --------------------"
        )

        for key in l2_norm_error_dict:
            create_l2_error_line_chart(l2_norm_error_dict[key], fold, output_folder, analysis_type, modeling_type, clip=key)

    # Generate paraview visualization for lowest and largest errors
    if errors_params.get("l2_error_in_sequence").get("paraview_largest_error"):        
        logger.info(
            "-----  Creating Paraview visualization for largest errors -------"
        )
        for key in l2_norm_error_dict:
            save_paraview_largest_errors(l2_norm_error_dict[key], ground_truth_df, prediction_df, spatial_indices, output_folder, fold, analysis_type, modeling_type, clip=key)
        
    if errors_params.get("l2_error_in_sequence").get("paraview_smallest_error"):        
        logger.info(
            "-----  Creating Paraview visualization for smallest errors -------"
        )
        for key in l2_norm_error_dict:
            save_paraview_smallest_errors(l2_norm_error_dict[key], ground_truth_df, prediction_df, spatial_indices, output_folder, fold, analysis_type, modeling_type, clip=key)
        

    # Move logging file to output folder
    if errors_params.get("copy_log"):        
        logger.info(
            "-------------------- Copying and pasting log error  --------------------"
        )
        copy_and_paste_log_file_directory(Path("."), output_folder)

