####################################################
# Benchmark: A collection of datasets-related methods
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
###################################################

from functools import lru_cache
import pickle
import sys
import os
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yaml
import argparse

# Points to the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "experiments", "data"))

@dataclass
class Configuration():
    """Configuration for a dataset."""
    name: int
    error_type: list
    labels: str
    ml_task: str
    dboost_configs: list
    fd_constraints: dict

    @staticmethod
    def load_config(dataset_name):
        """
        Loads the parameters.yml config for a given dataset and returns it as a
        Configuration class.

        Datasets should be located at ../datasets/. Exits if no config can be found.
        A configuration should have the following parameters:
            * k_clusters - integer
            * classifier - "DecisionTree", "NeuralNetwork"
            * zero_padding - True (zero-padding), False (delta-padding)
            (TODO check if zero-padding has a negative impact on Decision Tree)
            * profile - "Distribution", "StructureFeatures"
            * labeling - "ActiveLearning", "ClusteringLabelpropagation", "Clustering"
        """
        path = os.path.join(DATA_DIR, dataset_name, "parameters.yml")
        if not os.path.isfile(path):
            sys.exit(f"Error: No parameters.yml found for dataset '{dataset_name}'.")

        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return Configuration(
                config["name"],
                config["error_type"],
                config["labels"],
                config['ml_task'],
                config["fd_constraints"],
                config['dboost_configs'],
            )


def load_csv(dataset_path):
    """
    This method reads a dataset from a csv file path.

    Arguments:
    dataset_path -- string denoting the path of the dataset
    """

    # load data
    dataDF = pd.read_csv(dataset_path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)

    return dataDF

def store_csv(df, dataset_path):
    """
    stores given dataframe as .csv in dataset_path

    Arguments:
    df (dataframe) -- dataframe of the the input data
    dataset_path (String) -- path to the folder where the CSV file will be stored
    """
    df.to_csv(path, index=False, encoding="utf-8")

class Dataset:
    """Represents a dataset and its configuration."""
    name: str
    directory: str
    cfg: Configuration
    dirty_df: pd.DataFrame
    clean_df: pd.DataFrame

    def __init__(self, parameters):
        # Temporarily allow parameters to be a string (name) and dictionary (configuration)
        if isinstance(parameters, str):
            self.name = parameters
        else:
            self.name = parameters["name"]

        self.directory = os.path.abspath(os.path.join(DATA_DIR, self.name))
        self.cfg = Configuration.load_config(self.name)

        # Load dirty and clean dataframe
        self.clean_df = load_csv(os.path.join(self.directory, "clean.csv"))
        if os.path.exists(os.path.join(self.directory, "dirty.csv")):
            self.dirty_df = load_csv(os.path.join(self.directory, "dirty.csv"))


def store_actual_errors(actual_errors, error_rate, DATASET_PATH):
    """
    This method stores the actual errors into a csv file and the error rate into a json file

    :param actual_errors -- dictionay, indices of actual errors in a dataset
    :param error_rate -- float, the number of dirty cells relative to the total number of cells in a dataset
    :return:
    """

    # Create the path of the output files
    csv_path = os.path.join(DATASET_PATH, "actual_errors.csv")
    json_path = os.path.join(DATASET_PATH, "error_rate.json")

    with open(csv_path, "a", encoding="utf-8") as f_object:
        # Create a file object and prepare it for writing the results
        writefile = csv.writer(f_object)
        # Prepare the row which is to be written to the file
        for key, value in actual_errors.items():
            row = [key, value]
            # Write the values after flattening the row list obtained in the above line
            writefile.writerow(np.hstack(row))

    # Close the file object
    f_object.close()

    with open(json_path, "w") as handle:
        json.dump(error_rate, handle)

def get_actual_errors(dirty_df, ground_truth_df, DATASET_PATH):
    """
    This method estimates the actual errors in a dataset and the error rate

    Arguments:
    dirty_df (dataframe) -- dirty dataset
    ground_truth_df (dataframe) -- ground truth of the dataset

    Returns:
    actual_errors_dictionary (dictionary) -- keys represent i,j of dirty cells & values are constant string "DUUMY VALUE"
    error_reate -- error rate in dirtDF compared to groundtruthDF
    """

    # Create dictionary for the output
    actual_errors_dictionary = {}

    for col in dirty_df.columns:
        # Get the location of the next column
        col_j = dirty_df.columns.get_loc(col)

        for i, row in dirty_df.iterrows():

            try:
                if int(float(dirty_df.iat[i, col_j])) != int(float(ground_truth_df.iat[i, col_j])):
                    actual_errors_dictionary[(i, col_j)] = "DUMMY VALUE"
            except ValueError:
                if dirty_df.iat[i, col_j] != ground_truth_df.iat[i, col_j]:
                    actual_errors_dictionary[(i, col_j)] = "DUMMY VALUE"

    error_rate = len(actual_errors_dictionary) / ground_truth_df.size

    # Store the error rate and actual errors for later use
    store_actual_errors(actual_errors_dictionary, error_rate, DATASET_PATH)

    return actual_errors_dictionary, error_rate


if __name__ == "__main__":

    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset_name', nargs='+', default=None, required=True)

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset_name

    for dataset_name in dataset_names:
        data_obj = Dataset(dataset_name)

        # Prepare the paths
        dataset_path = os.path.abspath(os.path.join(DATA_DIR, dataset_name))
        clean_path = os.path.join(dataset_path, 'clean.csv')
        dirty_path = os.path.join(dataset_path, 'dirty.csv')

        # Load the dirty data and its ground truth
        dirty_df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", low_memory=False)
        clean_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)

        _, error_rate = get_actual_errors(dirty_df, clean_df, dataset_path)
        print(dataset_name, ': ', error_rate)

