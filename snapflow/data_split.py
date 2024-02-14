from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from snapflow.utils import logger, timing_decorator, check_parameters_and_extract
import numpy as np
import pandas as pd
from natsort import natsorted


class DataSplitter:
    def __init__(self, params):
        self.splitting_params = params["splitting"]
        self.random_state = params["random_state"]
        self.folded_data = {}

    def __distribute_split_data(self, data, splits):
        for i, (train_index, val_index) in enumerate(splits):
            logger.info(f'Splitting fold {i} of {len(splits)}')
            train_index = natsorted(train_index)
            val_index = natsorted(val_index)
            self.folded_data[i] = {}
            self.folded_data[i].update({"train": data.values[train_index, :].T})
            self.folded_data[i].update({"validation": data.values[val_index, :].T})
            self.folded_data[i].update({"train_indices": np.array(train_index)})
            self.folded_data[i].update({"validation_indices": np.array(val_index)})
        return

    def _temporal_split(self, data):
        # TODO: code WFCV from scratch
        data = pd.DataFrame(data)
        logger.info(
            "-------------------- Starting temporal cross validation strategy --------------------"
        )
        tscv = check_parameters_and_extract(self.splitting_params, "splitting_strategy")
        # tscv = TimeSeriesSplit(
        #     n_splits=self.splitting_params.get("number_of_folds_or_splits"),
        #     test_size=self.splitting_params.get("validation_size"),
        #     gap=self.splitting_params.get("gap"),
        # )
        logger.info(tscv)
        splits = list(tscv.split(data))
        logger.info(splits)
        self.__distribute_split_data(data, splits)
        return self.folded_data

    def _kfold_split(self, data):
        logger.info(
            "-------------------- Starting K-Fold cross validation strategy --------------------"
        )
        kf = KFold(
            n_splits=self.splitting_params.get("number_of_folds_or_splits"),
            shuffle=True,
            random_state=self.random_state,
        )
        splits = list(kf.split(data))
        self.__distribute_split_data(data, splits)
        return self.folded_data

    def _standard_split(self, data, size_type="validation_size", shuffle=True):
        if size_type == "test_size":
            logger.info(
                "-------------------- Starting standard train/test strategy --------------------"
            )
            (
                train_data,
                test_data,
            ) = train_test_split(
                data,
                test_size=self.splitting_params.get(size_type),
                shuffle=shuffle,
                random_state=self.random_state,
            )
            train_data.index = train_data.index.map(int)
            test_data.index = test_data.index.map(int)
            train_data.sort_index(inplace=True)
            test_data.sort_index(inplace=True)
            self.folded_data[0] = {}
            self.folded_data[0].update({"train_indices": train_data.T.columns})
            self.folded_data[0].update({"test_indices": test_data.T.columns})
            self.folded_data[0].update({"train": train_data.T.values})
            self.folded_data[0].update({"test": test_data.T.values})

        else:
            logger.info(
                "-------------------- Starting standard train/validation strategy --------------------"
            )
            train_data, val_data = train_test_split(
                data,
                test_size=self.splitting_params.get(size_type),
                shuffle=shuffle,
                random_state=self.random_state,
            )
            train_data.sort_index(inplace=True)
            val_data.sort_index(inplace=True)
            self.folded_data[0] = {}
            self.folded_data[0].update({"train": train_data.T})
            self.folded_data[0].update({"validation": val_data.T})
            self.folded_data[0].update({"train_indices": train_data.T.columns})
            self.folded_data[0].update({"validation_indices": val_data.T.columns})
        return self.folded_data

    def _preserve_test_data(self, data, shuffle=False):
        folded_data = self._standard_split(data, size_type="test_size", shuffle=shuffle)
        return folded_data

    def _null_split(self, data):
        self.folded_data[0].update({"train": data.T})
        self.folded_data[0].update({"validation": None})
        self.folded_data[0].update({"train_indices": data.T.columns})
        self.folded_data[0].update({"validation_indices": None})
        return self.folded_data

    def __assert_numpy_type(self, folded_data):
        for fold in folded_data.keys():
            for data_type in folded_data[fold].keys():
                try:
                    assert isinstance(
                        folded_data[fold][data_type], (np.ndarray, np.generic)
                    )
                except:
                    folded_data[fold][data_type] = folded_data[fold][data_type].values
        return folded_data

    def _log_run(self, folded_data):
        logger.info(f"Number of folds: {len(folded_data.keys())}")
        for key in folded_data:
            for inner_key in folded_data[key]:
                if folded_data[key][inner_key] is not None:
                    logger.info(
                        f"Dimensions of fold {key}: {inner_key}:  {folded_data[key][inner_key].shape}"
                    )
                    if "ind" in inner_key:
                        logger.info(
                            f"Indices for fold {key}: {inner_key}: {folded_data[key][inner_key]}"
                        )

    @timing_decorator
    def split_data(self, data, train_test_flag=False):
        column_names = [str(i) for i in range(data.shape[1])]
        data = pd.DataFrame(data=data, columns=column_names)
        data = data.T
        if train_test_flag:
            logger.info("Train/Test split selected.")
            folded_data = self._preserve_test_data(data)
            self._log_run(folded_data)
            return folded_data

        match self.splitting_params.get("splitting_strategy"):
            case "temporal":
                folded_data = self._temporal_split(data)
            case "kfold":
                folded_data = self._kfold_split(data)
            case "train_val":
                folded_data = self._standard_split(data)
            case _:
                folded_data = self._null_split(data)

        self._log_run(folded_data)
        folded_data = self.__assert_numpy_type(folded_data)
        return folded_data
