###################################################################
# ED2: implement the ED2 method to detect errors in a dataset

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import time
import pandas as pd
import os

from baseline.setup.detectors.ed2.model.ml.experiments.caller import run_ed2
from baseline.setup.utils import store_detections


def ed2(dirty_df, clean_path, dataset_name, label_cutoff, detections_path):
    """
    This method calls the ed2 algorithm to detect incorrect cells. Ed2 uses a active learning approach, thus the
    groundtruth dataset is needed (to simulate the human).

    Arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    dataset (String) -- name of the dataset
    configs (dict) -- "label_cutoff" specifies the budget of labels in the active learning process. The algoithm stop
        as soon as the number of labels exceeds the label_cutoff (in worst case it is 9 higher than the label_cutoff)

    Returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    evaluation_dict -- dictionary - hold evaluation information.
    """

    start_time = time.time()

    # transform dirtdf which has dtype string to numeric type if possible
    #dirty_df = dirty_df.apply(pd.to_numeric, errors="ignore")

    clean_df = pd.read_csv(
        os.path.abspath(clean_path),
        dtype=str,
        header="infer",
        encoding="utf-8",
        keep_default_na=False,
        low_memory=False,
    )

    # dataframe of same size as dirty_df that contains flase/true for every cell
    # true meaning, that the cell is an error. labels is the number of labels used in active learning
    all_error_statusDF, labels = run_ed2(clean_df, dirty_df, dataset_name, label_cutoff)

    detection_dictionary = {}
    for row_i in range(dirty_df.shape[0]):
        for col_i in range(dirty_df.shape[1]):
            if all_error_statusDF.iat[row_i, col_i]:
                detection_dictionary[(row_i, col_i)] = "JUST A DUMMY VALUE"

    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
