#############################################################################
# Detect: implement a method which executes different error detection methods

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
#############################################################################

import os
import pandas as pd

from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH
from baseline.setup.detectors.outlierDetector import outlierdetector
from baseline.setup.detectors.rahaDetector import raha
from baseline.setup.detectors.mvDetector import mvdetector
from baseline.setup.detectors.fahesDetector import fahes
from baseline.setup.detectors.kataraDetector import katara
from baseline.setup.detectors.nadeefDetector import nadeef
from baseline.setup.detectors.holoCleanDetector import holoclean
from baseline.setup.detectors.dBoostDetector import dboost
from baseline.setup.detectors.minKDetector import min_k
from baseline.setup.detectors.ed2Detector import ed2

from baseline.setup.utils import get_all_errors, create_detections_path
from baseline.model.utils import create_results_path, store_results_csv, ExperimentName, ExperimentType
from baseline.setup.evaluate import evaluate_detector


def detect(clean_path: str, 
           dirty_path: str, 
           detections_path: str, 
           dataset_path: str, 
           dataset_name: str, 
           detect_method: DetectMethod, 
           results_path: str, 
           mink_threshold: float = 0.4, 
           nb_labels: int = 20,
           split_ratio: float = 1.0):
    """
    Detect errors in a dataset using various detection methods
    """

    # Load the dirty data and its ground truth
    dirty_df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", dtype=str, low_memory=False)
    clean_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)
    
    # Initialize the metrics
    detection_time = 0
    precision = 0
    recall = 0
    f1_score = 0
    # Initialize a dictionary for storing the various parameters and metrics
    params_metrics_dict = {}

    if detect_method in [DetectMethod.OUTLIER_DETECTOR_IF, DetectMethod.OUTLIER_DETECTOR_SD,
                         DetectMethod.OUTLIER_DETECTOR_IQR]:
        method = detect_method.__str__()
        detections, detection_time = outlierdetector(dirtydf=dirty_df, 
                                                     detect_method=method, 
                                                     detections_path=detections_path)

    elif detect_method == DetectMethod.RAHA:
        detections, detection_time = raha(dataset_name=dataset_name, 
                                          clean_path=clean_path, 
                                          dirty_path=dirty_path,
                                          detections_path=detections_path,
                                          nb_labels=nb_labels)

    elif detect_method == DetectMethod.MV_DETECTOR:
        detections, detection_time = mvdetector(dirtydf=dirty_df, 
                                                detections_path=detections_path)

    elif detect_method == DetectMethod.FAHES_DETECTOR:
        detections, detection_time = fahes(dirtydf=dirty_df, 
                                           dirty_path=dirty_path, 
                                           detections_path=detections_path)

    elif detect_method == DetectMethod.KATARA:
        detections, detection_time = katara(dirtydf=dirty_df, 
                                            detections_path=detections_path)

    elif detect_method == DetectMethod.NADEEF:
        detections, detection_time = nadeef(dirty_df=dirty_df, 
                                            dataset_name=dataset_name, 
                                            detections_path=detections_path)

    elif detect_method == DetectMethod.HOLOCLEAN:
        detections, detection_time = holoclean(dirty_df=dirty_df, 
                                               dataset_name=dataset_name, 
                                               dataset_path=dataset_path, 
                                               detections_path=detections_path)

    elif detect_method == DetectMethod.DBOOST:
        detections, detection_time = dboost(dirty_df=dirty_df, 
                                            clean_df=clean_df,
                                            dataset_name=dataset_name, 
                                            detections_path=detections_path)

    elif detect_method == DetectMethod.MIN_K:
        detections, detection_time = min_k(dataset_name=dataset_name, 
                                           dirty_path=dirty_path, 
                                           clean_path=clean_path, 
                                           detections_path=detections_path, 
                                           threshold=mink_threshold)

    elif detect_method == DetectMethod.ED2_DETECTOR:
        label_cutoff = nb_labels * clean_df.shape[1]
        detections, detection_time = ed2(dirty_df=dirty_df, 
                                         clean_path=clean_path,
                                         dataset_name=dataset_name,
                                         label_cutoff=label_cutoff,
                                         detections_path=detections_path)

    else:
        raise NotImplemented
    
    if detections:
        # Evaluate detections, but first find all errors in the dataset
        all_errors = get_all_errors(dirty_df, clean_df, dataset_name)
        precision, recall, f1_score = evaluate_detector(all_errors=all_errors, detections=detections)
        
    else:
        print("No detections have been captured by the {} detector!".format(detect_method.__str__()))
        
     
    # Log the parameters and metrics
    params_metrics_dict.update( detector=detect_method.__str__(),
                                num_labels=nb_labels,
                                split_ratio=split_ratio,
                                precision=precision, 
                                recall=recall, 
                                f1_score=f1_score,
                                time=detection_time)

    # Store the results
    store_results_csv(params_metrics_dict, results_path)

    return detections, precision, recall, f1_score, detection_time


if __name__ == "__main__":
    # Get the data path
    dataset_name = 'nasa'
    method = DetectMethod.RAHA

    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
    dirty_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

    # Create a path to the detections.csv file
    detections_path = create_detections_path(EXP_PATH, dataset_name, method.__str__())

    detect(clean_path, dirty_path, detections_path, dataset_path, dataset_name, detect_method=method)
