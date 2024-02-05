###################################################################
# Raha: implement the RAHA method to detect errors in a dataset

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import time
import os
import logging

from baseline.setup.utils import store_detections
from baseline.setup.detectors.raha.raha.detection import Detection as Raha_Detection


def raha(dataset_name, clean_path, dirty_path, detections_path, nb_labels=20):
    start_time = time.time()

    # dict to process raha steps internally
    internal_dataset_dict = {
        "name": dataset_name,
        "path": dirty_path,
        "clean_path": clean_path
    }

    # detect errors and get detection dictionary
    app = Raha_Detection(nb_labels=nb_labels)
    detection_dictionary = app.run(internal_dataset_dict)

    # get runtime
    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
