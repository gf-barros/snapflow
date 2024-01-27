from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, params):
        self.splitting_params = params["splitting"]
        self.random_state = params["random_state"]
        self.folded_data = {}

    def _temporal_split(self, data):
        # TODO: code WFCV from scratch
        
        logger.info("-------------------- Starting temporal cross validation strategy --------------------")
        tscv = TimeSeriesSplit(n_splits=self.splitting_params.get("number_of_folds_or_splits"),
                               test_size=self.splitting_params.get("validation_size"),
                               gap=self.splitting_params.get("gap"))
        splits = list(tscv.split(data))
        for i, (train_index, val_index) in enumerate(splits):
            # logger.debug(f"Fold {i}:")
            # logger.debug(f"  Train: index={train_index}")
            # logger.debug(f"  Validation:  index={val_index}")
            self.folded_data[i] = {}
            self.folded_data[i].update({"train": data[train_index]})
            self.folded_data[i].update({"validation": data[val_index]})
        return self.folded_data
    
    def _kfold_split(self, data):
        logger.info("-------------------- Starting K-Fold cross validation strategy --------------------")
        kf = KFold(n_splits=self.splitting_params.get("number_of_folds_or_splits"), shuffle=True, random_state=self.random_state)
        splits = list(kf.split(data))

        for i, (train_index, val_index) in enumerate(splits):
            # logger.debug(f"Fold {i}:")
            # logger.debug(f"  Train: index={train_index}")
            # logger.debug(f"  Validation:  index={val_index}")
            self.folded_data[i] = {}
            self.folded_data[i].update({"train": data[train_index]})
            self.folded_data[i].update({"validation": data[val_index]})

        return self.folded_data

    def _standard_split(self, data, size_type="validation_size", shuffle=True):
        logger.info("-------------------- Starting standard train/validation strategy --------------------")
        if size_type == "test_size": 
            indices = np.arange(data.shape[1])
            (
                train_data,
                test_data,
                indices_train,
                indices_test,
            ) = train_test_split(data, indices, test_size=self.splitting_params.get(size_type), shuffle=shuffle, random_state=self.random_state)
            self.folded_data[0] = {}
            self.folded_data[0].update({"train": train_data})
            self.folded_data[0].update({"test": test_data})

        else:
            train_data, val_data= train_test_split(data, test_size=self.splitting_params.get(size_type), shuffle=shuffle, random_state=self.random_state)
            self.folded_data[0] = {}
            self.folded_data[0].update({"train": train_data})
            self.folded_data[0].update({"validation": val_data})
        return self.folded_data
    
    def preserve_test_data(self, data, shuffle=False):
        logger.info("-------------------- Starting standard train/test strategy --------------------")
        folded_data = self._standard_split(data, size_type="test_size", shuffle=shuffle)
        return folded_data
    
    def split_data(self, data):
        match self.splitting_params.get("strategy"):
            case "temporal":
                folded_data = self._temporal_split(data)
                logger.info(f"Number of folds: {len(folded_data.keys())}")
                for key in folded_data:
                    for inner_key in folded_data[key]:
                        logger.info(f"Dimensions of fold {key}: {folded_data[key][inner_key].shape}")

            case "kfold":
                folded_data = self._kfold_split(data)
                logger.info(f"Number of folds: {len(folded_data.keys())}")
                for key in folded_data:
                    for inner_key in folded_data[key]:
                        logger.info(f"Dimensions of fold {key}: {folded_data[key][inner_key].shape}")

            case "train_test":
                folded_data = self.preserve_test_data(data)
                logger.info(f"Number of folds: {len(folded_data.keys())}")
                for key in folded_data:
                    for inner_key in folded_data[key]:
                        logger.info(f"Dimensions of fold {key}: {folded_data[key][inner_key].shape}")

            case "train_val":
                folded_data = self._standard_split(data)
                logger.info(f"Number of folds: {len(folded_data.keys())}")
                for key in folded_data:
                    for inner_key in folded_data[key]:
                        logger.info(f"Dimensions of fold {key}: {folded_data[key][inner_key].shape}")


