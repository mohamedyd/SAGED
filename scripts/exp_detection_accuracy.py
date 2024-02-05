# Detection accuracy 
#   - Compare the performance of SAGED with respect to a number of baseline tools
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
from saged.utils import create_detections_path, EXP_PATH
from scripts.run_baseline import run_baselines
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_detection_accuracy import plot_exp_detection_accuracy


parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Define the parameters of the experiment 
dirty_datasets = ['beers', 'breast_cancer', 'flights', 'rayyan', 'bikes', 'smartfactory', 'nasa', 'soilmoisture', 'restaurants', 'hospital']
historical_datasets = ['adult', 'movies_1']
labeling_budgets = np.arange(start=5, stop=35, step=5)
involved_detectors = [DetectMethod.ED2_DETECTOR, DetectMethod.RAHA,
                      DetectMethod.KATARA, DetectMethod.DBOOST,
                      DetectMethod.FAHES_DETECTOR, DetectMethod.HOLOCLEAN,
                      DetectMethod.MIN_K, DetectMethod.OUTLIER_DETECTOR_SD,
                      DetectMethod.OUTLIER_DETECTOR_IQR, DetectMethod.OUTLIER_DETECTOR_IF]


exp_type = ExperimentName.DETECTION.__str__()

# Create an empty DataFrame to store the combined data
df_combined = pd.DataFrame()

for dataset_name in dirty_datasets:
    
    try:
        list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=dataset_name,
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
                                                                            runs=args.runs,
                                                                            num_labels=20,
                                                                            label_augmentation=None,
                                                                            verbose=args.verbose,
                                                                            exp_type=exp_type
                                                                            )
    
    
    
    except Exception as e:
        print(f"[ERROR] Failed to run SAGED")
        print("Exception: {}".format(e.args[0])) 
        continue
    
    # Run the baseline methods
    run_baselines(involved_detectors=involved_detectors, 
                    dataset_names=[dataset_name], 
                    num_labels=20, 
                    nb_runs=args.runs, 
                    experiment_type=exp_type, 
                    verbose=args.verbose)

# ======================= Generate a latex table of the results ==========================

    # Define the list of detectors involved in the experiment 
    detectors_list = [DetectMethod.FAHES_DETECTOR.__str__(), DetectMethod.OUTLIER_DETECTOR_IF.__str__(),
                        DetectMethod.OUTLIER_DETECTOR_IQR.__str__(), DetectMethod.OUTLIER_DETECTOR_SD.__str__(),
                        DetectMethod.KATARA.__str__(), DetectMethod.DBOOST.__str__(), DetectMethod.HOLOCLEAN.__str__(),
                        DetectMethod.MIN_K.__str__(),DetectMethod.RAHA.__str__(), DetectMethod.ED2_DETECTOR.__str__(),
                        DetectMethod.SAGED.__str__()]
    
    # Get the results path
    exp_name = ExperimentName.DETECTION.__str__()
    results_paths = []
    for method in detectors_list:
        results_paths.append(create_detections_path(EXP_PATH, dataset_name, method, exp_type=exp_name, store_detection_metrics=True))

    # Evaluation metrics
    eval_metrics = ['precision', 'recall', 'f1_score', 'time']
    # Generate a dataframe of the results for each dataset
    results_df = plot_exp_detection_accuracy(dataset=dataset_name, results_paths=results_paths, eval_metrics=eval_metrics)
    
    # Cobine the current DataFrame to the combined DataFrame
    df_combined = pd.concat([df_combined, results_df], axis=1)
    
# Create a latex table
latex_table = df_combined.to_latex(index=True, caption='Comparison of SAGED and baseline tools in terms of detection accuracy and runtime', column_format='lccc')

# Save the generated latex table
table_path = os.path.join(EXP_PATH, 'evaluation', 'plots', 'detection_table.tex')

with open(table_path, 'w') as f:
    f.write(latex_table)

if args.verbose:
    print("Table saved successfully!")