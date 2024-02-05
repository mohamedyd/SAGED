# Ablation study: similarity 
#   - Test the impact of the similairty tool used to find the most relevant historical datasets
#   - Find the performance of SAGED versus the number of historical datasets
# Author: Mohamed Abdelaal
# Date: May 2023

import argparse
from saged.saged import saged
from saged.utils import create_detections_path, EXP_PATH
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_similarity import plot_exp_similarity


parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Define the parameters of the experiment
similarities = ['cosine', 'clustering']
dirty_dataset = ['rayyan']
historical_datasets = ['adult', 'movies_1', 'hospital', 'flights', 'bikes', 'breast_cancer', 'restaurants']

for dataset_name in dirty_dataset:
    
    for iter in range(len(historical_datasets)):
        
        # Sample a subset of the historical datasets
        historical_datasets_iter = historical_datasets[:iter + 1]

        if args.verbose:
            print(f"Iter: {iter}")
            print(f"Historical datasets: {historical_datasets_iter}")
        
        for similairty_method in similarities:
            
            try:
                list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=dataset_name,
                                                                                    historical_datasets=historical_datasets_iter,
                                                                                    features="meta",
                                                                                    profile="structure_features",
                                                                                    classifier="mlp_classifier",
                                                                                    clustering="kmeans",
                                                                                    propagate_labels=False,
                                                                                    labeling="none",
                                                                                    similarity=similairty_method,
                                                                                    n_clusters=1,
                                                                                    n_meta_features=0,
                                                                                    runs=args.runs,
                                                                                    num_labels=20,
                                                                                    label_augmentation=None,
                                                                                    verbose=args.verbose,
                                                                                    exp_type=ExperimentName.SIMILARITY.__str__()
                                                                                    ) 

            except Exception as e:
                print(f"[ERROR] Failed to run SAGED with similarity {similairty_method} and num of historical datasets {iter}")
                print("Exception: {}".format(e.args[0])) 
                continue
    
    # ======================== Plot the results ==========================
    # Get the results path
    exp_name = ExperimentName.SIMILARITY.__str__()
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
    plot_exp_similarity(results_path=results_path,
                        plots_path=plots_path,
                        eval_metrics=eval_metrics,
                        y_labels=y_labels)