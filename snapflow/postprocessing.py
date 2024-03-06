import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import h5py
from snapflow.utils import logger, timing_decorator



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


class PostProcessing():
    def __init__(self, fold, ground_truth, predictions, indices, spatial_indices, params_dict, analysis_type="train", modeling_type="backtest", output_folder=None):
        """Instantiation of the PostProcessing class.

        Parameters
        ----------
        data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        svd_params_dict : dict
            Dictionary containing the SVD parameters.

        Attributes
        ----------
        self.u : numpy.ndarray
            A 2D array containing the left singular vectors modes from self.data.
        self.s : numpy.ndarray
            A 1D array containing the singular values from self.data.
        self.vt : numpy.ndarray
            A 2D array containing the right singular vectors modes from self.data.
        self.data : numpy.ndarray
            A 2D array containing the stacked snapshots.
        self.params_dict : dict
            A dictionary containing all the modeling parameters.
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.fold = fold
        self.indices = indices
        self.spatial_indices = spatial_indices
        self.analysis_type = analysis_type
        self.modeling_type = modeling_type
        self.postproc_params_dict = params_dict["postprocessing"]
        if output_folder:
            postproc_output_folder = output_folder / Path("postprocessing")
            if not os.path.exists(postproc_output_folder):
                os.mkdir(postproc_output_folder)
            self.output_folder = postproc_output_folder
                
    def __write_dict_to_csv(file_path, data_dict):
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as csvfile:
            fieldnames = data_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)

    def create_l2_error_line_chart(self, l2_norm_dict, clip=None):
        logger.info(
            "-------------------- Computing L2 error line chart --------------------"
        )

        max_index = max(l2_norm_dict, key=l2_norm_dict.get)
        max_value = l2_norm_dict[max_index]
        indices = list(l2_norm_dict.keys())
        values = list(l2_norm_dict.values())

        plt.plot(indices, values, label="Values")            
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"L2 norm error for fold {self.fold}")
        plt.axhline(
            y=max_value, color="red", linestyle="--", label=f"Max Value ({max_value:.2f})"
        )

        if self.output_folder:
            if clip is None: 
                plt.savefig(self.output_folder / Path(f"l2_error_line_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}.png"))
            else: 
                str_clip = str(clip).replace(".", "_")
                plt.savefig(self.output_folder / Path(f"l2_error_line_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}_clip_{str_clip}.png"))
        plt.close()

    def create_l2_error_bar_chart(self, l2_norm_dict, clip=None):
        logger.info(
                    "-------------------- Computing L2 error bar chart --------------------"
                )
        max_index = max(l2_norm_dict, key=l2_norm_dict.get)
        max_value = l2_norm_dict[max_index]
        indices = list(l2_norm_dict.keys())
        values = list(l2_norm_dict.values())

        plt.bar(indices, values, label="Values")           
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"L2 norm error for fold {self.fold}")
        plt.axhline(
            y=max_value, color="red", linestyle="--", label=f"Max Value ({max_value:.2f})"
        )

        if self.output_folder:
            if clip is None: 
                plt.savefig(self.output_folder / Path(f"l2_error_bar_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}.png"))
            else: 
                str_clip = str(clip).replace(".", "_")
                plt.savefig(self.output_folder / Path(f"l2_error_bar_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}_clip_{str_clip}.png"))
        plt.close()

    def copy_and_paste_log_file_directory(self, source_path, destination_path):
        logger.info(
            "-------------------- Copying and pasting log error  --------------------"
        )
        try:
            shutil.copy(
                source_path / Path("log_file.txt"), destination_path / Path("log_file.txt")
            )
        except:
            pass

    def compute_l2_norm_error(self, ground_truth_df, prediction_df):
        l2_norm_error = {}
        for column in tqdm(ground_truth_df.columns):
            diff = ground_truth_df[column] - prediction_df[column]
            norm_diff = np.linalg.norm(diff)
            norm_original = np.linalg.norm(ground_truth_df[column])

            # Avoid division by zero
            if norm_original != 0:
                l2_norm_error[column] = norm_diff / norm_original
            else:
                l2_norm_error[column] = np.nan  # or some other placeholder value indicating undefined error

        return l2_norm_error

    def compute_clipped_l2_norm_error(self, ground_truth_df, prediction_df, clip=None):
        l2_norm_error = {}
        for column in tqdm(ground_truth_df.columns):
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

    def save_paraview_largest_errors(self, l2_norm_errors, ground_truth_df, prediction_df):
        logger.info(
                    "-----  Creating Paraview visualization for largest errors -------"
                )
        # Finding the key with the largest error
        key_largest_error = max(l2_norm_errors, key=l2_norm_errors.get)
        ground_truth_largest_error = ground_truth_df.loc[self.spatial_indices, key_largest_error].values
        prediction_largest_error = prediction_df.loc[self.spatial_indices, key_largest_error].values

        print(ground_truth_largest_error.shape)
        print(ground_truth_largest_error.dtype)
        print(type(ground_truth_largest_error))

        # Saving solutions for largest error
        save_paraview_visualization(
                    ground_truth_largest_error,
                    self.output_folder,
                    f"ground_truth_largest_error_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}_{key_largest_error}",
                )
        
        save_paraview_visualization(
                    np.real(prediction_largest_error),
                    self.output_folder,
                    f"prediction_largest_error_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}_{key_largest_error}")

    def save_paraview_smallest_errors(self, l2_norm_error_dict, ground_truth_df, prediction_df):
        logger.info(
            "-----  Creating Paraview visualization for smallest errors -------"
        )
        # Finding the key with the smallest error
        key_smallest_error = min(l2_norm_error_dict, key=l2_norm_error_dict.get)
        ground_truth_smallest_error = ground_truth_df.loc[self.spatial_indices, key_smallest_error].values
        prediction_smallest_error = prediction_df.loc[self.spatial_indices, key_smallest_error].values

        # Saving solutions for smallest error
        save_paraview_visualization(
                    ground_truth_smallest_error,
                    self.output_folder,
                    f"ground_truth_smallest_error_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}_{key_smallest_error}")
        
        save_paraview_visualization(
                    np.real(prediction_smallest_error),
                    self.output_folder,
                    f"prediction_smallest_error_{self.modeling_type}_{self.analysis_type}_fold_{self.fold}_{key_smallest_error}")

    def compute_frobenius_norm_error(self):
        output_dict = {}
        if self.output_folder:
            general_output_folder = self.output_folder / Path("general_outputs")
            if not os.path.exists(general_output_folder):
                os.mkdir(general_output_folder)

        # Frobenius norm
        if self.postproc_params_dict.get("frobenius_norm"):
            logger.info("-------------------- Computing Frobenius norm--------------------")
            frobenius_norm = np.linalg.norm(self.predictions - self.ground_truth) / np.linalg.norm(
                self.ground_truth
            )
            output_dict[self.fold] = {}
            output_dict[self.fold]["analysis_type"] = self.analysis_type
            output_dict[self.fold]["frobenius_rel_error"] = frobenius_norm
            dict_path = general_output_folder / Path(
                f"general_{self.modeling_type}_{self.analysis_type}.csv"
            )
            self.__write_dict_to_csv(dict_path, output_dict)

    def compute_l2_norm_error_in_sequence(self):
            logger.info(
                "-------------------- Computing L2 Norm across all data --------------------"
            )
            print(self.spatial_indices.shape)
            ground_truth_df = pd.DataFrame(self.ground_truth, columns=self.indices, index=self.spatial_indices)
            prediction_df = pd.DataFrame(self.predictions, columns=self.indices, index=self.spatial_indices)

            l2_norm_error_dict = {}

            clips = self.postproc_params_dict.get("l2_error_in_sequence").get("clip")
            if clips is None:
                l2_norm_error_dict["no_clip"] = self.compute_l2_norm_error(ground_truth_df, prediction_df) 
            else:
                for clip in clips:
                    l2_norm_error_dict[clip] = self.compute_clipped_l2_norm_error(ground_truth_df, prediction_df, clip=clip) 
            return l2_norm_error_dict, ground_truth_df, prediction_df

    

    @timing_decorator
    def compute_errors(
        self
    ):
        logger.info(
            "-------------------- Computing Errors and Metrics --------------------"
        )

        # Compute Frobenius Norm
        self.compute_frobenius_norm_error()

        # L2 norm across all data
        if self.postproc_params_dict.get("l2_error_in_sequence").get("active"):
            l2_norm_errors, ground_truth_df, prediction_df = self.compute_l2_norm_error_in_sequence()

            # Plot L2 bar chart across all data
            if self.postproc_params_dict.get("l2_error_in_sequence").get("bar_plot_error"):
                for key in l2_norm_errors:
                    self.create_l2_error_bar_chart(l2_norm_errors[key])

            # Plot L2 line chart across all data
            if self.postproc_params_dict.get("l2_error_in_sequence").get("line_plot_error"):
                for key in l2_norm_errors:
                    self.create_l2_error_line_chart(l2_norm_errors[key])

            # Generate paraview visualization for lowest and largest errors
            if self.postproc_params_dict.get("l2_error_in_sequence").get("paraview_largest_error"):        
                for key in l2_norm_errors:
                    self.save_paraview_largest_errors(l2_norm_errors[key], ground_truth_df, prediction_df)
                
            if self.postproc_params_dict.get("l2_error_in_sequence").get("paraview_smallest_error"):        
                for key in l2_norm_errors:
                    self.save_paraview_smallest_errors(l2_norm_errors[key], ground_truth_df, prediction_df)
            

        # Move logging file to output folder
        if self.postproc_params_dict .get("copy_log"):        
            self.copy_and_paste_log_file_directory(Path("."), self.output_folder)

