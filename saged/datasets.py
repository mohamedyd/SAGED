"""Dataset handling."""

import os
import sys
from functools import lru_cache
from pathlib import Path
import pickle
import pandas as pd
import pathlib

# Points to ../datasets
# DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), "..", "datasets")
EXP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
DATASETS_DIR = os.path.abspath(os.path.join(EXP_DIR, "data"))


class Dataset:
    """Represents a dataset and its configuration.

    A dataset in the scope of this project consists of two files, dirty.csv (from the real world)
    and clean.csv (ground truth), which are located in ../datasets in a directory with the name
    of the dataset. These files are automatically loaded as pandas dataframes when creating a
    Dataset object. This class also contains methods for saving and loading classifier models in
    the dataset's directory (under a subfolder classifiers/).
    """
    name: str
    directory: str
    dirty_df: pd.DataFrame
    clean_df: pd.DataFrame

    def __init__(self, name):
        self.name = name
        self.directory = os.path.abspath(os.path.join(DATASETS_DIR, self.name))

        # Check if clean and dirty data exists and, if yes, load the dataframes
        dirty_path = os.path.join(self.directory, "dirty.csv")
        
        if not os.path.exists(dirty_path):
            raise FileNotFoundError(f"Couldn't find file 'dirty.csv' for dataset {self.name}.")

        clean_path = os.path.join(self.directory, "clean.csv")
        if not os.path.exists(dirty_path):
            raise FileNotFoundError(f"Couldn't find file 'clean.csv' for dataset {self.name}.")

        self.dirty_df = pd.read_csv(dirty_path, low_memory=False)
        self.clean_df = pd.read_csv(clean_path, low_memory=False)

        # Make sure that the labels are identical
        self.dirty_df.columns = self.clean_df.columns

    def __str__(self):
        # <name> (<rows>, <columns>)
        return f"{self.name} {self.dirty_df.shape}"

    @staticmethod
    def load_all(skip=None):
        """Load all available datasets in the datasets/ directory. Prints a warning if a dataset
        couldn't be loaded, but does not exit.

        Parameters:
            skip (list[str], optional): Datasets to skip when loading. Defaults to None.

        Returns:
            list[Dataset]: List of loaded datasets.
        """
        if skip is None:
            skip = []

        datasets = []

        for filename in os.listdir(DATASETS_DIR):
            if os.path.isdir(os.path.join(DATASETS_DIR, filename)) and filename not in skip:
                try:
                    datasets.append(Dataset(filename))
                except FileNotFoundError as ex:
                    print(f"Warning: {str(ex)} (Dataset will not be loaded.)")

        return datasets

    def save_classifier_model(self, classifier, classifier_type, column):
        """Saves a classifier model to a pickle file. The path to the pickle file is
        datasets/<name>/models/<classifier-type>/clf-<column-name>.pkl.

        Parameters:
            classifier (general classifier): The classifier model to save.
            classifier_type (saged.configuration.ClassifierType): Type of classifier to save.
            column (str): Name of the column for the model.
        """
        # Create path to directory first and check if it exists
        #dir_path = Path(os.path.join(DATASETS_DIR, self.name, "models", classifier_type.value))
        dir_path = Path(os.path.abspath(os.path.join(EXP_DIR, 'evaluation', 'data', self.name, "models", classifier_type.value)))

        if not dir_path.exists():
            pathlib.Path(dir_path).mkdir(parents=True)
            #dir_path.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(dir_path, "clf-" + column + ".pkl"), "wb") as f:
            pickle.dump(classifier, f)

    @lru_cache
    def load_classifier_model(self, classifier_type, column):
        """Loads a classifier model from a pickle file. This function is cached: multiple calls with
        the same parameters will load the classifier model from memory.

        Parameters:
            classifier_type (saged.configuration.ClassifierType): Type of classifier to load.
            column (str): Name of the column for the model.

        Raises:
            FileNotFoundError: If the file for the classifier model can't be found.

        Returns:
            General classifier: A general classifier (type determined by classifier_type)
        """
        # Classifiers are stored at: datasets/<name>/models/<classifier-type>/clf-<column-name>.pkl
        model_path = os.path.abspath(os.path.join(EXP_DIR, 'evaluation', 'data', self.name,
                                                  "models", classifier_type.value, "clf-" + column + ".pkl"))
        #model_path = os.path.join(
        #    DATASETS_DIR, self.name, "models", classifier_type.value, "clf-" + column + ".pkl"
        #)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Couldn't find models for dataset {self.name} (expected file {model_path})"
            )
        
        print(model_path)
        with open(model_path, "rb") as f:
            return pickle.load(f)

    @lru_cache
    def get_actual_errors(self):
        """Counts the actual errors (cached).

        :return numpy.array: 2d array with 1s at coordinates with errors
        """
        return (self.dirty_df != self.clean_df).astype(int)
        #return (self.dirty_df.where(self.dirty_df.values != self.clean_df.values).notna())
