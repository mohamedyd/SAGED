# modeling accuracy 
#   - Compare the performance of SAGED with respect to a number of baseline tools in terms of modeling accuracy while using GT as a repair tool
#   - Baselines: raha, ed2, SD, IQR, IF, min-k, fahes, holoclean, dboost, katara
#
# Author: Mohamed Abdelaal
# Date: May 2023

import os
import argparse
import pandas as pd
import numpy as np
from sys import exit
from saged.saged import saged
from saged.utils import create_detections_path, create_target_path, EXP_PATH, DATA_PATH
from scripts.run_baseline import run_baselines
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_detection_accuracy import plot_exp_detection_accuracy
from baseline.setup.repairs.repair import RepairMethod
from saged.train_e2e_saged import train_e2e_saged
from baseline.model.train import train_model
from baseline.baseline import train_e2e_baseline


parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Define the parameters of the experiment 
#dirty_datasets = ['beers', 'breast_cancer', 'bikes', 'soilmoisture', 'smartfactory', 'adult']
dirty_datasets = ['breast_cancer']
historical_datasets = ['adult', 'movies_1']

exp_type = ExperimentName.MODELING.__str__()

involved_detectors = [DetectMethod.ED2_DETECTOR, DetectMethod.RAHA,
                      DetectMethod.KATARA, DetectMethod.DBOOST,
                      DetectMethod.FAHES_DETECTOR, DetectMethod.HOLOCLEAN,
                      DetectMethod.MIN_K, DetectMethod.OUTLIER_DETECTOR_SD,
                      DetectMethod.OUTLIER_DETECTOR_IQR, DetectMethod.OUTLIER_DETECTOR_IF]

baseline_only = False
involved_detectors = [DetectMethod.RAHA]
involved_repairs = [RepairMethod.ML_IMPUTER]


for dataset_name in dirty_datasets:
    
    if not baseline_only:
        try:
            for repair_method in involved_repairs:
                
                # ======================== SAGED Pipeline =====================================
                # Prepare the paths
                dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
                clean_path = os.path.join(dataset_path, 'clean.csv')
                dirty_path = os.path.join(dataset_path, 'dirty.csv')
                detections_path = create_detections_path(EXP_PATH, dataset_name, DetectMethod.SAGED.__str__())
                det_results_path = create_detections_path(EXP_PATH, dataset_name, DetectMethod.SAGED.__str__(), store_detection_metrics=True)
                target_path = create_target_path(EXP_PATH, dataset_name, DetectMethod.SAGED.__str__(), repair_method.__str__())

                for _ in range(args.runs):

                    train_e2e_saged(clean_path=clean_path,
                                    dirty_path=dirty_path,
                                    detections_path=detections_path,
                                    target_path=target_path,
                                    repair_method=repair_method,
                                    dirty_dataset=dataset_name,
                                    historical_datasets=historical_datasets,
                                    features="meta",
                                    profile="structure_features",
                                    classifier="mlp_classifier",
                                    clustering="kmeans",
                                    propagate_labels=False,
                                    labeling='none',
                                    similarity='clustering',
                                    n_clusters=1,
                                    n_meta_features=0,
                                    runs=1,
                                    num_labels=20,
                                    label_augmentation=None,
                                    verbose=args.verbose,
                                    exp_type=exp_type,
                                    tune_params=False,
                                    epochs=500)
        
        
        except Exception as e:
            print(f"[ERROR] Failed to run the E2E-SAGED pipeline")
            print("Exception: {}".format(e)) 
            continue
        
        # ==================== Clean Pipeline ==================================
        
        # Retrieve the dirty and clean data
        clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))

        # Load the dirty data and its ground truth
        data_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)

        # Train a model
        experiment_name = ExperimentName.MODELING.__str__()
        experiment_type = ExperimentType.GROUND_TRUTH.__str__()

        for _ in range(args.runs):

            train_model(data_df,
                        dataset_name,
                        tune_params=False,
                        exp_name=experiment_name,
                        exp_type=experiment_type,
                        verbose=args.verbose,
                        epochs=500)

    # ======================== Baseline Pipeline =============================
    
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

            for _ in range(args.runs):

                try:
                    train_e2e_baseline(clean_path=clean_path,
                                        dirty_path=dirty_path,
                                        detections_path=detections_path,
                                        detection_results_path=det_results_path,
                                        target_path=target_path,
                                        detect_method=detect_method,
                                        repair_method=repair_method,
                                        dataset_name=dataset_name,
                                        dataset_path=dataset_path,
                                        verbose=args.verbose,
                                        tune_params=False,
                                        epochs=500,
                                        nb_labels=20)
                except Exception as e:
                    print(e)
                    print("[ERROR] Failed to run the Baseline pipeline")
                    break

    