###################################################################
# Min_k: implement the min-k ensemble method to detect errors in a dataset

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import time
import os
import pandas as pd
from baseline.setup.utils import store_detections
from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH
from baseline.setup.utils import get_all_errors, create_detections_path
from baseline.setup.evaluate import evaluate_detector



def min_k(dataset_name, dirty_path, clean_path, detections_path, threshold=0.2):
    """
    Run at the very end, after all other non-ensembledetectors ran.
    Considers those errors that are detected by at least configs["threshold"] percent of detectors (only stand alone detectors).

    Arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    dataset (String) -- name of the dataset
    configs (dict) -- has to contain "threshold" which specifies the minimum percentage of how
    many detectors need to detect an error in order to be included

    Returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    evaluation_dict -- dictionary - hold evaluation information.
    """

    start_time = time.time()

    detectors_list = list(DetectMethod)

    detection_dicts = []
    for detector_name in detectors_list:
        if detector_name.__str__() not in ['min_k', 'baseline']:
            path = create_detections_path(EXP_PATH, dataset_name, detector_name.__str__(), create_new_dirs=False)
            # Check whether the detections file exist
            if os.path.exists(path):
                try:
                    reader = pd.read_csv(path, names=['i', 'j', 'dummy'])
                    detection_dicts.append(reader.groupby(['i', 'j'])['dummy'].apply(list).to_dict())
                except:
                    continue
            else:
                continue
        else:
            continue

    # for each detected cell count the number of times it was detected over all detection_dicts
    cells_counter = {}
    for i, detections in enumerate(detection_dicts):
        for cell in detections.keys():
            if cell not in cells_counter:
                cells_counter[cell] = 0.0
            cells_counter[cell] += 1.0

    # for each detected error get percentage of detectors that detected the error
    for cell in cells_counter:
        cells_counter[cell] /= len(detection_dicts)

    # fill detection_dictionary with detections that have been detected
    # by a minimum of threshold-percent of the detectors
    detection_dictionary = {}
    for cell in cells_counter:
        if cells_counter[cell] >= threshold:
            detection_dictionary[cell] = "JUST A DUMMY VALUE"

    error_detect_runtime = time.time( ) -start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
