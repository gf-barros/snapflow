
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path
import shutil
import h5py
from src.utils import logger


def write_dict_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = data_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)


def create_l2_error_bar_chart(l2_norm_dict, fold, output_folder, analysis_type="train"):
    indices = list(l2_norm_dict.keys())
    values = list(l2_norm_dict.values())
    max_index = values.index(max(values))
    max_value = values[max_index]

    plt.bar(indices, values, label='Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'L2 norm error for fold {fold}')
    plt.axhline(y=max_value, color='red', linestyle='--', label=f'Max Value ({max_value:.2f})')

    for i, v in enumerate(values):
        plt.text(indices[i], v + 0.1, f'{v:.2f}', ha='center', va='bottom')

    #plt.legend()

    if output_folder:
        plt.savefig(output_folder / Path(f"l2_error_{analysis_type}_fold_{fold}.png"))
    plt.close()


def copy_and_paste_paraview_directory(source_path, destination_path):
    try:
        shutil.copy(source_path / Path('concentration_1.h5'), destination_path / Path('concentration_1.h5'))
        shutil.copy(source_path / Path('concentration_1.xdmf'), destination_path / Path('concentration_1.xdmf'))
    except:
        pass

def copy_and_paste_log_file_directory(source_path, destination_path):
    try:
        shutil.copy(source_path / Path('log_file.txt'), destination_path / Path('log_file.txt'))
    except:
        pass

def insert_h5_vector(vector, paraview_output_folder):
    """Injects vector on H5 files for post-processing viz."""
    filename_output = Path(os.path.join(
        paraview_output_folder,
        "concentration_1.h5",
    ))
    with h5py.File(filename_output, "r+") as h5_file_output:
        h5_file_output["concentration"]["concentration_0"]["vector"][...] = vector[
            :, np.newaxis
        ]

def save_paraview_visualization(l2_norm_error, indices, ground_truth, prediction, output_folder, plots="max"):
    paraview_input_folder = Path(os.path.join("data/visualization"))
    if output_folder:
        paraview_output_folder = output_folder / Path("paraview_plots")
        if not os.path.exists(paraview_output_folder):
            os.mkdir(paraview_output_folder)

    if plots == "max":
        max_error_index = max(l2_norm_error.items(), key=lambda x: x[1])[0]
        max_error_list_index = list(indices).index(max_error_index)
        export_dict = {
            "ground_truth": ground_truth[:, max_error_list_index],
            "prediction": prediction[:, max_error_list_index],
            "absolute_error": np.abs(ground_truth[:, max_error_list_index] - prediction[:, max_error_list_index])
        }
        for export in export_dict.keys():
            paraview_output_export_folder = paraview_output_folder / Path(f'{export}_{max_error_index}')
            if not os.path.exists(paraview_output_export_folder):
                os.mkdir(paraview_output_export_folder)
            copy_and_paste_paraview_directory(paraview_input_folder, paraview_output_export_folder)
            logger.info(export)
            logger.info(export_dict[export].shape)            
            insert_h5_vector(export_dict[export], paraview_output_export_folder)




def compute_errors(fold, prediction, ground_truth, indices, output_folder, analysis_type="train", modeling_type="backtest"):
    logger.info("-------------------- Computing Errors and Metrics --------------------")
    logger.info(ground_truth.shape)
    logger.info(prediction.shape)

    output_dict = {}
    if output_folder:
        general_output_folder = output_folder / Path("general_outputs")
        if not os.path.exists(general_output_folder):
            os.mkdir(general_output_folder) 

    # Frobenius norm
    logger.info("-------------------- Computing Frobenius norm--------------------")
    frobenius_norm = np.linalg.norm(prediction - ground_truth)/np.linalg.norm(ground_truth)
    output_dict[fold] = {}
    output_dict[fold]["analysis_type"] = analysis_type
    output_dict[fold]["frobenius_rel_error"] = frobenius_norm
    dict_path = general_output_folder / Path(f'general_{modeling_type}_{analysis_type}.csv')
    write_dict_to_csv(dict_path, output_dict)

    # L2 norm across all data
    logger.info("-------------------- Computing L2 Norm across all data --------------------")
    l2_norm_error = {}
    entry = 0
    for index in indices:
        l2_norm_error[index] = np.linalg.norm(prediction[entry, :] - ground_truth[entry, :], 2)/np.linalg.norm(ground_truth[entry, :], 2)
        entry += 1
    

    # Plot L2 norm across all data
    logger.info("-------------------- Computing L2 error bar chart --------------------")
    create_l2_error_bar_chart(l2_norm_error, fold, output_folder, analysis_type)

    # Create Paraview visualization across all (or any) data
    logger.info("-------------------- Computing Paraview visualization --------------------")
    save_paraview_visualization(l2_norm_error, indices, ground_truth, prediction, output_folder, plots="max")
    
    # Move logging file to output folder
    copy_and_paste_log_file_directory(Path("."), output_folder)


