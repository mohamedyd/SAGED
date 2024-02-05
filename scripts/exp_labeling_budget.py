# Labeling budget 
#   - Test the impact of the labeling budget of SAGED, RAHA, and ED2
#   - Find the performance of SAGED, RAHA, and ED2 versus the number of user-provided labels
# Author: Mohamed Abdelaal
# Date: May 2023

import argparse
import numpy as np
from sys import exit
from saged.saged import saged
from saged.utils import create_detections_path, EXP_PATH
from scripts.run_baseline import run_baselines
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_labeling_budget import plot_exp_labeling_budget


parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Define the parameters of the experiment 
dirty_datasets = ['beers', 'flights', 'bikes', 'nasa', 'soilmoisture', 'smartfactory', 'restaurants', 'hospital']
historical_datasets = ['adult', 'movies_1']
labeling_budgets = np.arange(start=5, stop=35, step=5)
involved_detectors = [DetectMethod.ED2_DETECTOR, DetectMethod.RAHA]
exp_type = ExperimentName.LABELING_BUDGET.__str__()


for dataset_name in dirty_datasets:
    
    for budget in labeling_budgets:
        
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
                                                                                num_labels=budget,
                                                                                label_augmentation=None,
                                                                                verbose=args.verbose,
                                                                                exp_type=exp_type
                                                                                )
        
        
        
        except Exception as e:
            print(f"[ERROR] Failed to run SAGED with labeling budget {budget}")
            print("Exception: {}".format(e.args[0])) 
            continue
        
        # Run the baseline methods
        run_baselines(involved_detectors=involved_detectors, 
                      dataset_names=[dataset_name], 
                      num_labels=budget, 
                      nb_runs=args.runs, 
                      experiment_type=exp_type, 
                      verbose=args.verbose)
        
# ======================= Plot the results ===========================================

    # Define a list of detectors
    detectors_list = [DetectMethod.RAHA.__str__(), DetectMethod.ED2_DETECTOR.__str__(), DetectMethod.SAGED.__str__()]
    
    # Get the results path
    results_paths = []
    for method in detectors_list:
        results_paths.append(create_detections_path(EXP_PATH, dataset_name, method, exp_type=exp_type, store_detection_metrics=True))


    plots_path = []
    # Get a path to the accuracy figure
    fig_name = 'accuracy' + '_' + dataset_name + '.pdf'
    plots_path.append(create_results_path(dataset_name=dataset_name,
                                            experiment_name=exp_type,
                                            filename=fig_name,
                                            parent_dir='plots'))

    # Get a path to the execution time figure
    fig_name = 'time' + '_' + dataset_name + '.pdf'
    plots_path.append(create_results_path(dataset_name=dataset_name,
                                            experiment_name=exp_type,
                                            filename=fig_name,
                                            parent_dir='plots'))


    # Plot the results of Aug2Clean and the clean dataset
    y_labels = ['F1 Score', 'Detection time (S)']
    eval_metrics = ['f1_score', 'time']
    plot_exp_labeling_budget(results_paths=results_paths,
                                plots_path=plots_path,
                                eval_metrics=eval_metrics,
                                y_labels=y_labels)