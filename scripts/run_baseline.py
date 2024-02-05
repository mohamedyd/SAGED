############################################################
# Detect errors in dirty datasets using the baseline methods
# Authors: Mohamed Abdelaal
# Date: March 2023
# Software AG
# All Rights Reserved
#############################################################

import os
import mlflow
import argparse
import statistics
from numpy import mean
from baseline.setup.detectors.detect_method import DetectMethod, DATA_PATH, EXP_PATH
from baseline.setup.utils import create_detections_path
from baseline.setup.detectors.detect import detect
from baseline.model.utils import ExperimentName


def run_baselines(involved_detectors: list, 
                  dataset_names: list, 
                  num_labels: int = 20, 
                  nb_runs: int = 1, 
                  experiment_type: str = ExperimentName.DETECTION.__str__(), 
                  verbose: str = True, 
                  run_mlflow: bool = False,
                  split_ratio: float = 1.0):

    if run_mlflow:
        # Start an MLflow experiment
        mlflow.set_experiment("Baseline")

    for dataset_name in dataset_names:

        # Initialize metrics lists
        list_precision = []
        list_recall = []
        list_f1_score = []
        list_total_time = []

        for detect_method in involved_detectors:

            # Prepare the paths
            dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
            clean_path = os.path.join(dataset_path, 'clean.csv')
            dirty_path = os.path.join(dataset_path, 'dirty.csv')

            detections_path = create_detections_path(EXP_PATH, dataset_name, detect_method.__str__(), exp_type=experiment_type)
            # Prepare a path to store the results
            results_path = create_detections_path(exp_path=EXP_PATH, 
                                                    data_name=dataset_name, 
                                                    detector_name=detect_method.__str__(),
                                                    exp_type=experiment_type,
                                                    store_detection_metrics=True) 
            
            for _ in range(nb_runs):
                
                # Detect errors
                if verbose:
                    print("====================================================")
                    print("=================== Running Baseline ===============")
                    print("====================================================\n")
                try:
                    _, det_precision, det_recall, det_f1_score, det_time = detect(clean_path=clean_path, 
                                                                                    dirty_path=dirty_path, 
                                                                                    detections_path=detections_path, 
                                                                                    dataset_path=dataset_path, 
                                                                                    dataset_name=dataset_name, 
                                                                                    detect_method=detect_method,
                                                                                    results_path=results_path,
                                                                                    nb_labels=num_labels,
                                                                                    split_ratio=split_ratio)
                    # Append the obtained results
                    list_precision.append(det_precision)
                    list_recall.append(det_recall)
                    list_f1_score.append(det_f1_score)
                    list_total_time.append(det_time)
                    
                    if verbose:
                        print(f"Precision:{det_precision}, Recall:{det_recall}, F1 Score:{det_f1_score}, Detection time:{det_time}")
                    
                except Exception as e:
                    print("[ERROR] Failed to run the error detection step")
                    print("Exception: {}".format(e.args))
                    continue

            if run_mlflow:
                # Log relevant information about the experiment
                with mlflow.start_run():
                    # log model parameters
                    mlflow.log_param("Dirty Dataset", args.dirty_dataset)
                    mlflow.log_param("Detector", args.detection_method)
                    mlflow.log_param("Number of Runs", args.runs)
                    mlflow.set_tag("Explanation", args.tags)
                        # log model metrics
                    mlflow.log_metric("Average Precision", mean(list_precision))
                    mlflow.log_metric("Average Recall", mean(list_recall))
                    mlflow.log_metric("Average F1 Score", mean(list_f1_score))
                    mlflow.log_metric("Average Total Time", mean(list_total_time))
                    if args.runs > 1:
                        mlflow.log_metric("Precision (Std)", statistics.stdev(list_precision))
                        mlflow.log_metric("Recall (Std)", statistics.stdev(list_recall))
                        mlflow.log_metric("F1 Score (Std)", statistics.stdev(list_f1_score))
                        mlflow.log_metric("Total Time (Std)", statistics.stdev(list_total_time))
                    
                    
if __name__ == '__main__':
    
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    # Add the parser's options
    parser.add_argument('--dirty-dataset',  nargs='+', default=None, required=True)
    parser.add_argument('--detection-method',  nargs='+', type=DetectMethod, choices=list(DetectMethod), default=None)
    parser.add_argument("--num-labels", type=int, default=20, help="number of labels to use")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument("--tags", type=str, default="None")
    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dirty_dataset
    verbose = args.verbose
    detectors_list = args.detection_method
    nb_runs = args.runs

    # Create a list of all available detectors and repair methods
    available_detectors = list(DetectMethod)

    # Use all available base detectors if no specific detectors are selected
    involved_detectors = available_detectors if not detectors_list else detectors_list
    if verbose:
        print("[INFO] The involved detectors are: {}".format(involved_detectors))
        
    run_baselines(involved_detectors=involved_detectors,
                  dataset_names=dataset_names, 
                  num_labels=args.num_labels, 
                  nb_runs=nb_runs, 
                  verbose=verbose)
