##########################################################
# mvDetector: implement the NADEEF error detection method

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
##++######################################################

import time
import re
import sys
import pandas as pd
from baseline.setup.utils import store_detections
from baseline.dataset.dataset import Dataset


def nadeef(dirty_df, dataset_name, detections_path):
    """
    This method runs NADEEF.
    It will return an empty dictionary and result dictionary if there are not constraints defined for the given dataset.

    Arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset

    Returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    """

    # fill na with dummy string to avoid errors due to nan
    dirty_df.fillna('0', axis=0, inplace=True)

    start_time = time.time()

    # Define a data object to get the list of labels in the dataset
    data_object = Dataset(dataset_name)

    fd_constraints = data_object.cfg.fd_constraints

    # define a dictionary to store the indices of the detected dirty cells
    detection_dictionary = {}

    pattern_violation_count = 0
    fd_violation_count = 0

    # return empty detection and results dict if dataset has no nadeef constraints
    if not fd_constraints:
        print("NADEEF: No FD constraints have been provided!")
        return {}, {}

    # adds (index, left_value) and (index, right_value) to dictionary for every functional dependency
    for fd in fd_constraints["functions"]:

        # get attribute of interest
        l_attribute, r_attribute = fd

        # get values of each attribute
        l_j = dirty_df.columns.get_loc(l_attribute)
        r_j = dirty_df.columns.get_loc(r_attribute)

        value_dictionary = {}

        # fills value dictionary with {value_left_i : {value_right_i : 1} value_left_i : {value_right_i : 1} ... }
        #
        for i, row in dirty_df.iterrows():
            if row[l_attribute]:
                if row[l_attribute] not in value_dictionary:
                    value_dictionary[row[l_attribute]] = {}
                if row[r_attribute]:
                    value_dictionary[row[l_attribute]][row[r_attribute]] = 1

        for i, row in dirty_df.iterrows():
            if (
                    row[l_attribute] in value_dictionary
                    and len(value_dictionary[row[l_attribute]]) > 1
            ):
                detection_dictionary[(i, l_j)] = "JUST A DUUMY VALUE"
                detection_dictionary[(i, r_j)] = "JUST A DUUMY VALUE"
                # increment fd violation by two for each pair of row, attribute_left and row attribute_right
                fd_violation_count = fd_violation_count + 2

    for attribute, pattern, opcode in fd_constraints["patterns"]:
        j = dirty_df.columns.get_loc(attribute)
        for i, value in dirty_df[attribute].iteritems():
            if opcode == "OM":
                if len(re.findall(pattern, value, re.UNICODE)) > 0:
                    detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
                    # increase pattern violation count for every pattern violation
                    pattern_violation_count = pattern_violation_count + 1
            else:
                if len(re.findall(pattern, value, re.UNICODE)) == 0:
                    detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
                    # increase pattern violation count for every pattern violation
                    pattern_violation_count = pattern_violation_count + 1

    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
