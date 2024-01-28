""" Functions for building features for surrogate modeling """

import h5py
from src.utils import logger
import numpy as np
from natsort import natsorted
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import os


def read_h5_libmesh(filename, dataset):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    h5_file = h5py.File(filename, "r")
    data = h5_file[(dataset)]
    data_array = np.array(data, copy=True)
    h5_file.close()
    return data_array


def read_h5_fenics(filename):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    with h5py.File(filename, "r") as h5_file_input:
        data_array = h5_file_input["concentration"]["concentration_0"]["vector"][...]
    return data_array


def snapshots_assembly(snapshots_params):
    """
    Function used to assembly snapshots matrix from csv, h5 or vtk files.

    Parameters
    ----------
    file_type_str : str
        String describing the simulations output files type.
        Possible inputs:
            - h5_libmesh: libMesh/EdgeCFD HDF5 files.
            - h5_fenics: FEniCS HDF5 files.
    snapshot_ingestion_parameters : dict
        Dictionary containing the information regarding the files.
        Keys:
            - filenames: List[str] (common to all possible file_type_str)
                List of strings containing the files to be ingested.
                Eg. ["/home/file001.h5", "/home/file002.h5"]
            - dataset: str (exclusive to h5_libmesh)
                String informing the key where the data will be read.
                Eg. "pressure"

    Returns
    -------
    snapshots : np.array
        Numpy 2D array containing the snapshots matrix.
    """
    logger.info("Starting choice of file type:")
    file_type_str = snapshots_params["file_type_str"]

    if file_type_str == "h5_libmesh":
        logger.info("libMesh/EdgeCFD HDF5 file selected.")
        dataset = snapshots_params["dataset"]
        ingestion_function = read_h5_libmesh
        ingestion_parameters = [filenames[0], dataset]
    elif file_type_str == "h5_fenics":
        logger.info("FEniCS HDF5 file selected.")
        filenames = []
        for contain_files in snapshots_params["file_name_contains"]:
            logger.info(f"Searching for files that containg the string '{contain_files}'.")
            for dirpath, _, file_names in os.walk(snapshots_params["folder"]):
                for f in file_names:
                    if f.endswith(".h5"): # and contain_files in f:
                        filenames.append(os.path.join(dirpath, f))
        logger.info(f"{len(filenames)} snapshots found.")
        filenames = natsorted(filenames)
        ingestion_function = read_h5_fenics
        ingestion_parameters = [filenames[0]]

    first_snapshot = ingestion_function(*ingestion_parameters)
    rows = first_snapshot.shape[0]
    columns = len(filenames)
    snapshots = np.zeros((rows, columns))
    snapshots[:, 0] = np.squeeze(first_snapshot)
    progress_bar = tqdm(range(1, columns))
    for i in progress_bar:
        progress_bar.set_description("Loading Snapshots %d" % i)
        ingestion_parameters[0] = filenames[i]
        data = ingestion_function(*ingestion_parameters)
        snapshots[:, i] = np.squeeze(data)
    return filenames, snapshots


def data_normalization(data, params, pipeline_stage, transpose=False):
    if transpose:
        data = data.T
    if params["normalization"][pipeline_stage] == "min_max":
        normalization_technique_class = MinMaxScaler()
        transformed_data = normalization_technique_class.fit_transform(data)
    if params["normalization"][pipeline_stage]  == "standard_scaler":
        normalization_technique_class = StandardScaler()
        transformed_data = normalization_technique_class.fit_transform(data)
    if transpose:
        transformed_data = transformed_data.T
    return transformed_data, normalization_technique_class



