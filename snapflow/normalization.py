""" Functions for normalizing data for surrogate modeling """
from snapflow.utils import logger, timing_decorator
from sklearn.preprocessing import MinMaxScaler, StandardScaler



@timing_decorator
def data_normalization(data, params, pipeline_stage, transpose=False):
    if transpose:
        data = data.T
    if params["normalization"][pipeline_stage] == "min_max":
        normalization_technique_class = MinMaxScaler()
        transformed_data = normalization_technique_class.fit_transform(data)
    if params["normalization"][pipeline_stage] == "standard_scaler":
        normalization_technique_class = StandardScaler()
        transformed_data = normalization_technique_class.fit_transform(data)
    if transpose:
        transformed_data = transformed_data.T
    return transformed_data, normalization_technique_class
