###############################################################################
# Utils: implement a set of helping functions used by repair and detect methods

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###############################################################################

import numpy as np
import pandas as pd
import csv
import os
from enum import Enum 
import pathlib

EXP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
DATA_PATH = os.path.abspath(os.path.join(EXP_PATH, "data"))

class DetectMethod(Enum):
    SAGED = 'saged'

    def __str__(self):
        return self.value
    

def create_detections_path(exp_path, data_name, detector_name, exp_type='detection', 
                           store_detection_metrics=False, create_new_dirs=True):
    """
    Creates a path to the detections.csv file
    """
    # Create a directory for such experiments, called detection
    eval_detector_path = os.path.abspath(os.path.join(exp_path, 'evaluation', 'data', data_name, exp_type,
                                                      detector_name))
    if not os.path.exists(eval_detector_path) and create_new_dirs:
        pathlib.Path(eval_detector_path).mkdir(parents=True)

    if store_detection_metrics:
        path = os.path.abspath(os.path.join(eval_detector_path, 'results.csv'))
    else:
        path = os.path.abspath(os.path.join(eval_detector_path, 'detections.csv'))

    return path


def create_target_path(exp_path, dataset_name, detector_name, repair_name):
    """
    Creates a path to the repaired.csv file
    """
    # Create a directory for such experiments, called repair
    eval_repair_path = os.path.abspath(os.path.join(exp_path, 'evaluation', 'data', dataset_name, 'repair',
                                                    detector_name, repair_name))
    if not os.path.exists(eval_repair_path):
        pathlib.Path(eval_repair_path).mkdir(parents=True)

    target_path = os.path.abspath(os.path.join(eval_repair_path, 'repaired.csv'))

    return target_path


def load_detections(detection_path):
    """
    Load the indices of the detected dirty instances
    @arguments:
    detection_path -- string, path to the CSV file containing the detection dictionary
    @return:
    detection_dict -- dictionary of the retrieved detections
    """
    try:
        reader = pd.read_csv(os.path.join(detection_path), names=['i', 'j', 'dummy'])
        detection_dict = reader.groupby(['i', 'j'])['dummy'].apply(list).to_dict()

    except Exception as e:
        print("Exception: {}".format(e.args[0]))
        return {}

    # If detections found, return a dictionary
    return detection_dict


def store_detections(dirty_instances, detection_path):
    """
    Store the dictionary of all dirty instances in a dataset

    @arguments:
     dirty_instances -- dictionay, indices of dirty instances in a dataset
     detection_path -- string, path to the detection dictionary
    """
    
    with open(detection_path, 'w') as f_object:
        # Create a file object and prepare it for writing the results
        writefile = csv.writer(f_object)
        # Prepare the row which is to be written to the file
        for key, value in dirty_instances.items():
            row = [key, value]
            # Write the values after flattening the row list obtained in the above line
            writefile.writerow(np.hstack(row))

    # Close the file object
    f_object.close()


def get_all_errors(dirty_df, clean_df, dataset_name='None'):
    """
    Find all dirty instances in a dataset using the ground truth and return a dictionary of their indices

    @arguments:
    dirtyDF (dataframe) -- dirty dataset
    groundTruthDF (dataframe) -- ground truth of the dataset

    @return:
    dirty_instances_dictionary -- dictionary, indices i,j of dirty instances in a dataset
    """

    # Create dictionary for the output
    dirty_instances_dictionary = {}

    # Convert dirty data into numeric, since it was loaded as strings
    dirty_df = dirty_df.apply(pd.to_numeric, errors="ignore")

    # Iterate over each column and each row in the dirty dataset
    for col in dirty_df.columns:
        # Get the location of the next column
        col_j = dirty_df.columns.get_loc(col)

        for i, row in dirty_df.iterrows():
            # Compare the corresponding cells in dirty and clean datasets
            if dirty_df.iat[i, col_j] != clean_df.iat[i, col_j]:
                dirty_instances_dictionary[(i, col_j)] = "DUMMY VALUE"

    error_rate = len(dirty_instances_dictionary) / clean_df.size
    print(dataset_name, error_rate)

    return dirty_instances_dictionary


def store_results_csv(results, target_path):
    """Store the results in CSV files"""

    # Check whether the file already exists: useful for writing the header only once
    file_exists = os.path.isfile(target_path)

    # Store the results
    with open(target_path, 'a', newline='') as f_object:
        w = csv.DictWriter(f_object, results.keys())

        if not file_exists:
            # Write the header
            w.writeheader()

        # write the results
        w.writerow(results)

        # Close the file object
        f_object.close()