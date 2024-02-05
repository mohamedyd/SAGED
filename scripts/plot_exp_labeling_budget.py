# Plot: labeling budget
#   - Test the impact of the labeling budget on RAHA, SAGED, and ED2
#   - Find the performance versus the number of user-provided labels
# Author: Mohamed Abdelaal
# Date: May 2023


import argparse
import matplotlib
import pandas as pd
import numpy as np
from sys import exit
import matplotlib.pyplot as plt
from saged.utils import create_detections_path, EXP_PATH
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.model.utils import create_results_path, ExperimentName, ExperimentType

# ========== Plot configurations =======
plt.rc('grid', linestyle=":", color='black', alpha=0.3)
bar_width = 0.3
bar_width_runtime = 0.9
opacity = 0.4
error_config = {'ecolor': '0.3'}
matplotlib.rcParams.update({'font.size': 16})
colors = ["g", "b", "r", "y", "c", "m", "k", "orange"]
colors2 = ["red", "green"]
patterns = ["/", "o", "\\", "x", "|", ".", "*", "-", "+", "O", '/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']

labels_mapper = {
    'raha': 'RAHA',
    'ed2': 'ED2',
    'saged': 'SAGED',
}

def plot_exp_labeling_budget(results_paths, plots_path, eval_metrics, y_labels):
    """Plot detection accuracy versus the number of historical datasets"""

    # Read the CSV files
    labels_list = []
    means_dict = {}
    variances_dict = {}
    for results_path in results_paths:
        results = pd.read_csv(results_path, low_memory=False)

        results_gb = results.groupby(['num_labels'])
    
        # Store the results of each detector in a dictionary
        means_dict[results['detector'][0]] = results_gb[eval_metrics].mean()
        variances_dict[results['detector'][0]] = results_gb[eval_metrics].std()
        
        # Estimate the keys 
        results_keys = results_gb.groups.keys()
        # Append the labels
        labels_list.append(results['detector'][0])
        
    # Extract x_values via finding the number of historical datasets
    x_axis_values = list(results_keys)
        
    for plot_path, eval_metric, y_label in zip(plots_path, eval_metrics, y_labels):  

        # Define list of markers
        markers_list = ['.', '+', 'x', 'o', '*', '8', 's', 'd', '1', '2', '3', '4', '|']

        fig, ax = plt.subplots()

        for idx, imethod in enumerate(labels_list):
            
            metric = means_dict[method][eval_metric]
            variance = variances_dict[method][eval_metric]  
            
            plt.plot(x_axis_values, list(metric), marker=markers_list[idx], linewidth=3, markersize=8,
                    alpha=0.6, label=labels_mapper[method])
            plt.fill_between(x_axis_values, [x - y for x,y in zip(metric, variance)], [x + y for x,y in zip(metric, variance)],
                            alpha=0.3, antialiased=True)

        plt.xlabel('Labeling Budget')
        plt.grid(True)
        plt.ylabel(y_label)
        plt.legend(loc='best', ncol=2)

        # Save the figure
        plt.savefig(plot_path, dpi=1200, orientation='portrait', format='pdf', bbox_inches='tight')
        plt.tight_layout()
        
        

if __name__ == '__main__':

    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset', nargs='+', default=None, required=True)

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset

    for dataset_name in dataset_names:
        
        # Define the list of detectors involved in the experiment 
        detectors_list = [DetectMethod.RAHA.__str__(), DetectMethod.ED2_DETECTOR.__str__(), DetectMethod.SAGED.__str__()]
        
        # Get the results path
        exp_name = ExperimentName.LABELING_BUDGET.__str__()
        results_paths = []
        for method in detectors_list:
            results_paths.append(create_detections_path(EXP_PATH, dataset_name, method, exp_type=exp_name, store_detection_metrics=True))


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
        plot_exp_labeling_budget(results_paths=results_paths,
                                   plots_path=plots_path,
                                   eval_metrics=eval_metrics,
                                   y_labels=y_labels)