#####################################################################################
# Fahes: implement the FAHES detector to detect outliers and disguised missing values

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
#####################################################################################

import time
import pandas as pd
from baseline.setup.detectors.FAHES.fahes_caller import executeFahes
from baseline.setup.utils import store_detections


def fahes(dirtydf, dirty_path, detections_path, detect_method='ALL'):
    """
    This method detects disguised missing values.

    In order to run this method, fahes has to be compiled. Therefore go to cleaners/FAHES/src run "make clean" and then "make all".

    Arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    dataset (String) -- name of the dataset
    path_to_dirtydf (String) -- path to the dirty dataframe, will be used by fahes.
    tool (String) -- which fahes component is to be run. SYN-OD = check syntactic outliers only;
                 RAND = detect DMVs that replace MAR values; NUM-OD = detect DMVs that are numerical outliers only;
                 ALL = check all DMVs

    Returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    evaluation_dict -- dictionary - hold evaluation information.
    """

    start_time = time.time()

    # dict to map modules that fahes should run and its respective identifier
    module_dict = {
        "SYN-OD": 1,
        "RAND": 2,
        "NUM-OD": 3,
        "ALL": 4
    }
    # run fahes and get path to results .csv
    path_fahes_res = executeFahes(dirty_path, module_dict[detect_method])

    # load results .csv as dataframe
    fahes_res_df = pd.read_csv(
        path_fahes_res,
        dtype=str,
        header="infer",
        encoding="utf-8",
        keep_default_na=False,
        low_memory=False,
    )

    detection_dictionary = {}

    # for each entry in fahes results go through the respective
    # column in dirtydf and mark every cell as detected that
    # has the DMV value defined in the fahes results entry
    for i_fahes, row_fahes in fahes_res_df.iterrows():
        for j_dirty, row_dirty in dirtydf.iterrows():
            # get index of respective column in dirtdf
            col_index = dirtydf.columns.get_loc(row_fahes["Attribute Name"])
            if row_dirty[row_fahes["Attribute Name"]] == row_fahes["DMV"]:
                detection_dictionary[(j_dirty, col_index)] = "JUST A DUMMY VALUE"

    # get runtime
    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
