###################################################
# dBoost: implement the dBoost error detector

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################

import time
import os
import hashlib
import json
import itertools
import random
import tempfile
import pandas as pd
import sys

from baseline.dataset.dataset import Dataset
from baseline.setup.evaluate import evaluate_detector
from baseline.setup.utils import store_detections, get_all_errors
from baseline.setup.detectors.dBoost.dboost.imported_dboost import run_dboost


def dboost(dirty_df, clean_df, dataset_name, detections_path):
    """
    This method detects outliers with dboost. It runs fast when pre-estimated configurations exist in the YAML file

    @arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    dataset (String) -- name of the dataset

    @returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    evaluation_dict -- dictionary - hold evaluation information.
    """
    start_time = time.time()
    algorithm_and_configurations = []
    algorithm = "OD"

    # Define a dataset object
    data_object = Dataset(dataset_name)
    configs = data_object.cfg.dboost_configs

    # Use the predefined configs, if defined in the dataset dictionary or in the configurations JSON file
    if configs:
        configuration_list = [configs]
    else:
        # list of configurations to run dboost mit
        configuration_list = [
            list(a) for a in
            list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                   ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
            list(itertools.product(["gaussian"], ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
        random.shuffle(configuration_list)

    # run each strategy and save the evaluation results and detection dictionary
    # of the strategy with the highest f1 score
    best_f1 = -1.0
    best_strategy_profile = {}
    for config in configuration_list:
        outputted_cells = {}
        strategy_name = json.dumps([algorithm, config])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))

        # create a tmp directory and write the dirty dataframe to it
        dataset_path = os.path.join(tempfile.gettempdir(), dataset_name + "-" + strategy_name_hash + ".csv")
        dirty_df.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

        # run dboost with respective parameters of configuration
        params = ["-F", ",", "--statistical", "0.5"] + ["--" + config[0]] + config[1:] + [dataset_path]
        run_dboost(params)

        # get results from dboost and create detection dictionar (outputted cells) of it
        # the remove the tmp directory
        algorithm_results_path = dataset_path + '-dboost_output.csv'
        if os.path.exists(algorithm_results_path):
            ocdf = pd.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                               keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
            for i, j in ocdf.values.tolist():
                if int(i) > 0:
                    outputted_cells[(int(i) - 1, int(j))] = "JUST A DUMMY VALUE"
            os.remove(algorithm_results_path)
        os.remove(dataset_path)

        # evaluate strategy save as best strategy it f1 score is so far the highest
        all_errors = get_all_errors(dirty_df, clean_df)
        precision, recall, f1 = evaluate_detector(all_errors, outputted_cells)

        if f1 > best_f1:
            best_strategy_profile = {"name": strategy_name, "detection_dict": outputted_cells, "precision": precision,
                                     "recall": recall, "f1": f1}
            best_config = config
            best_f1 = f1

    error_detect_runtime = time.time() - start_time
    detection_dictionary = best_strategy_profile["detection_dict"]

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
