# Ablation study: labeling strategy 
#   - Test the impact of the labeling strategies used to find the most relevant tuples to be labeled by an oracle
#   - Compare four labeling strategies, including random, heuristic, active learning, and clustering
#   - Find the performance of SAGED versus the number of user-provided labels
# Author: Mohamed Abdelaal
# Date: May 2023

import argparse
import numpy as np
from sys import exit
from saged.saged import saged
from saged.utils import create_detections_path, EXP_PATH
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_labeling_strategy import plot_exp_labeling_strategy


parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Define the parameters of the experiment 
dirty_datasets = ['beers', 'breast_cancer', 'flights', 'hospital', 'rayyan', 'soilmoisture', 'smartfactory']
historical_datasets = ['adult', 'movies_1']
labeling_strategies = ['none', 'clustering', 'active_learning', 'heuristic']
labeling_budgets = np.arange(start=5, stop=35, step=5)

for dataset_name in dirty_datasets:
    
    for budget in labeling_budgets:
        
        for strategy in labeling_strategies:
            
            try:
                list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=dataset_name,
                                                                                    historical_datasets=historical_datasets,
                                                                                    features="meta",
                                                                                    profile="structure_features",
                                                                                    classifier="mlp_classifier",
                                                                                    clustering="kmeans",
                                                                                    propagate_labels=False,
                                                                                    labeling=strategy,
                                                                                    similarity='clustering',
                                                                                    n_clusters=1,
                                                                                    n_meta_features=0,
                                                                                    runs=args.runs,
                                                                                    num_labels=budget,
                                                                                    label_augmentation=None,
                                                                                    verbose=args.verbose,
                                                                                    exp_type=ExperimentName.LABELING_STRATEGY.__str__()
                                                                                    )
            
            except Exception as e:
                print(f"[ERROR] Failed to run SAGED with labeling strategy {strategy} and labeling budget {budget}")
                print("Exception: {}".format(e.args[0])) 
                continue
            
    # ================ Plot the results ======================== 

    # Get the results path
    exp_name = ExperimentName.LABELING_STRATEGY.__str__()
    results_path=create_detections_path(EXP_PATH, dataset_name, "saged", exp_type=exp_name, store_detection_metrics=True)


    plots_path = []
    # Get a path to the accuracy figure
    fig_name = 'accuracy' + '_' + dataset_name + '.pdf'
    plots_path.append(create_results_path(dataset_name=dataset_name,
                                            experiment_name=exp_name,
                                            filename=fig_name,
                                            parent_dir='plots'))

    # Get a path to the execution time figure
    fig_name = 'time' + '_' + dataset_name + '.pdf'
    plots_path.append(create_results_path(dataset_name=dataset_name,
                                            experiment_name=exp_name,
                                            filename=fig_name,
                                            parent_dir='plots'))


    # Plot the results of Aug2Clean and the clean dataset
    y_labels = ['F1 Score', 'Detection time (S)']
    eval_metrics = ['f1_score', 'time']
    plot_exp_labeling_strategy(results_path=results_path,
                                plots_path=plots_path,
                                eval_metrics=eval_metrics,
                                y_labels=y_labels) 