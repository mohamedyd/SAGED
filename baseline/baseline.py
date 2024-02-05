########################################
# Baseline: Implement a baseline method
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
########################################

import os
from baseline.model.train import train_model
from baseline.setup.detectors.detect_method import DATA_PATH, EXP_PATH
from baseline.setup.detectors.detect import detect, DetectMethod
from baseline.setup.repairs.repair import repair, RepairMethod
from baseline.setup.utils import create_target_path, create_detections_path
from baseline.model.utils import ExperimentType, ExperimentName


def train_e2e_baseline(clean_path,
                       dirty_path,
                       detections_path,
                       detection_results_path,
                       target_path,
                       detect_method,
                       repair_method,
                       dataset_name,
                       dataset_path,
                       verbose=True,
                       tune_params=True,
                       epochs=500,
                       exp_name=ExperimentName.MODELING.__str__(),
                       error_rate=0,
                       nb_labels=20):

    """
    Train a pipeline with error detector, data repair method, and keras models
    """

    # Detect errors
    if verbose:
        print("====================================================")
        print("=================== Error Detection ================")
        print("====================================================\n")
    try:
        _, det_precision, det_recall, det_f1_score, det_time = detect(clean_path, 
                                                                      dirty_path, 
                                                                      detections_path, 
                                                                      dataset_path, 
                                                                      dataset_name, 
                                                                      results_path=detection_results_path,
                                                                      detect_method=detect_method,
                                                                      nb_labels=nb_labels)
    except Exception as e:
        print("[ERROR] Failed to run the error detection step")
        print("Exception: {}".format(e.args[0]))
        raise FileNotFoundError
    # Repair errors
    if verbose:
        print("====================================================")
        print("=================== Error Repair ===================")
        print("====================================================")
        print(f"Repairing the {dataset_name} dataset using the {repair_method.__str__()} repair method...", end="", flush=True)
    repaired_df = repair(clean_path, dirty_path, target_path, detections_path, dataset_name, repair_method)
    if verbose:
        print("done.")

    # Train a model
    if verbose:
        print("====================================================")
        print("=================== Model Training =================")
        print("====================================================\n")

    exp_type = ExperimentType.E2E_PIPELINE.__str__() + '_' + detect_method.__str__() + '_' + repair_method.__str__()
    train_model(data_df=repaired_df,
                data_name=dataset_name,
                tune_params=tune_params,
                exp_type=exp_type,
                exp_name=exp_name,
                verbose=verbose,
                epochs=epochs,
                error_rate=error_rate)


if __name__ == '__main__':
    # Get the data path
    dataset_name = 'adult'
    detect_method = DetectMethod.RAHA
    repair_method = RepairMethod.ML_IMPUTER

    # Prepare the paths
    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    clean_path = os.path.join(dataset_path, 'clean.csv')
    dirty_path = os.path.join(dataset_path, 'dirty.csv')
    detections_path = create_detections_path(EXP_PATH, dataset_name, detect_method.__str__())
    target_path = create_target_path(EXP_PATH, dataset_name, detect_method.__str__(), repair_method.__str__())

    train_e2e_baseline(clean_path, dirty_path, detections_path, target_path, detect_method, repair_method,
                       dataset_name, dataset_path, verbose=True, tune_params=False)
