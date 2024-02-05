####################################################################################
# mvDetector: implement missing value detector method

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################################

import time
from baseline.setup.utils import store_detections


def mvdetector(dirtydf, detections_path):
    """
    This method detects explicit missing values.

    As the data is loaded with keep_default_na = False, empty cells are interpreted as
    empty string. Thus every cell with an empty string is counted as a explicit missing value.

    Arguments:
    dirtyDF -- dataframe, dirty version of a dataset
    detections_path -- string, path to store the detections

    Returns:
    detection_dictionary -- dictionary, keys represent i,j of dirty cells & values are string "JUST A DUUMY VALUE"
    """
    start_time = time.time()

    isnull_matrix = dirtydf.isnull()

    detection_dictionary = {}

    # save every (row,col) where entry is true to detection dictionary
    for col in dirtydf.columns:

        col_j = dirtydf.columns.get_loc(col)

        for i, row in dirtydf.iterrows():

            # dataset is read with keep_default_na = False, so empty cells
            # are represented by empty strings
            if dirtydf.iat[i, col_j] == '' or isnull_matrix.iat[i, col_j]:
                detection_dictionary[(i, col_j)] = "JUST A DUMMY VALUE"

    # get runtime
    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)


    return detection_dictionary, error_detect_runtime

