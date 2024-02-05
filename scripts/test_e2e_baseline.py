######################################################################
# Train ML models on data repaired using different cleaning strategies
# Authors: Mohamed Abdelaal
# Date: April 2022
# Software AG
# All Rights Reserved
######################################################################

import os
import argparse
from baseline.setup.detectors.detect_method import DetectMethod, DATA_PATH, EXP_PATH
from baseline.setup.repairs.repair import RepairMethod
from baseline.setup.utils import create_target_path, create_detections_path
from baseline.baseline import train_e2e_baseline


# Initialize an argument parser
parser = argparse.ArgumentParser()
# Add the parser's options
parser.add_argument('--dirty-dataset',  nargs='+', default=None, required=True)
parser.add_argument('--detection-method',  nargs='+', type=DetectMethod, choices=list(DetectMethod), default=None)
parser.add_argument('--repair-method',  nargs='+', type=RepairMethod, choices=list(RepairMethod), default=None)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--tune-hyperparams', action='store_true')
parser.add_argument('--nb-iterations', type=int, default=1)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument("--num-labels", type=int, default=20, help="number of labels to use")
args = parser.parse_args()

# Retrieve the input arguments
dataset_names = args.dirty_dataset
verbose = args.verbose
detectors_list = args.detection_method
repair_list = args.repair_method
nb_iterations = args.nb_iterations
epochs = args.epochs
tune_hyperparams = args.tune_hyperparams

# Create a list of all available detectors and repair methods
available_detectors = list(DetectMethod)
available_repair = list(RepairMethod)

# Use all available base detectors if no specific detectors are selected
involved_detectors = available_detectors if not detectors_list else detectors_list
involved_repairs = available_repair if not repair_list else repair_list
if verbose:
    print("[INFO] The involved detectors are: {}".format(involved_detectors))
    print("[INFO] The involved repair methods are: {}".format(involved_repairs))

for dataset_name in dataset_names:

    for repair_method in involved_repairs:

        for detect_method in involved_detectors:

            # Prepare the paths
            dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
            clean_path = os.path.join(dataset_path, 'clean.csv')
            dirty_path = os.path.join(dataset_path, 'dirty.csv')
            detections_path = create_detections_path(EXP_PATH, dataset_name, detect_method.__str__())
            det_results_path = create_detections_path(EXP_PATH, dataset_name, detect_method.__str__(), store_detection_metrics=True)
            target_path = create_target_path(EXP_PATH, dataset_name, detect_method.__str__(),
                                                repair_method.__str__())

            for _ in range(nb_iterations):

                #try:
                train_e2e_baseline(clean_path=clean_path,
                                    dirty_path=dirty_path,
                                    detections_path=detections_path,
                                    detection_results_path=det_results_path,
                                    target_path=target_path,
                                    detect_method=detect_method,
                                    repair_method=repair_method,
                                    dataset_name=dataset_name,
                                    dataset_path=dataset_path,
                                    verbose=verbose,
                                    tune_params=tune_hyperparams,
                                    epochs=epochs,
                                    nb_labels=args.num_labels)
                #except Exception as e:
                #    print(e)
                #    print("[ERROR] Failed to run the pipeline")
                #    break

                    