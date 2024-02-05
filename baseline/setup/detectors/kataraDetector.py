####################################################################################
# mvDetector: implement the KATARA method to detect inconsistencies and outliers

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################################

import time
import os
import random

from baseline.setup.detectors.katara.katara import run_KATARA
from baseline.setup.utils import store_detections


def katara(dirtydf, detections_path):
    """
    This method detects cells that violate the knowledge base in cleaner/KATARA/knowledge-base.

    Arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    detections_path -- string, path to the detections

    Returns:
    detection_dictionary -- dictionary, keys represent i,j of dirty cells & values are string "JUST A DUUMY VALUE"
    """

    start_time = time.time()

    # get list of Relations from knowledge base to run KATARA mit
    path_to_knowledge = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, "katara", "knowledge-base"))
    configuration_list = [os.path.join(path_to_knowledge, pat) for pat in os.listdir(path_to_knowledge)]
    random.shuffle(configuration_list)

    detection_dictionary = {}

    # fill detection_dictionary with detections based on different relations of knowledge-base
    for config in configuration_list:
        outputted_cells = run_KATARA(dirtydf, config)
        detection_dictionary.update({cell: "JUST A DUMMY VALUE" for cell in outputted_cells})

    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
