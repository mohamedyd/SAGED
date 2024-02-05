# Plot scalability analysis 
#   - Compare the scalability of SAGED with respect to a number of baseline tools
#   - Baselines: raha, ed2, SD, IQR, IF, min-k, fahes, holoclean, dboost, katara
#
# Author: Mohamed Abdelaal
# Date: May 2023

import os
import argparse
import matplotlib
import pandas as pd
import numpy as np
from sys import exit
import matplotlib.pyplot as plt
from saged.utils import create_detections_path, EXP_PATH
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.model.utils import create_results_path, ExperimentName, ExperimentType

labels_mapper = {
    'raha': 'RAHA',
    'ed2': 'ED2',
    'saged': 'SAGED',
    'katara': 'KATARA',
    'min_k': 'Min-k',
    'fahes': 'FAHES',
    'holoclean': 'HoloClean',
    'SD': 'SD',
    'IQR': 'IQR',
    'IF': 'IF',
    'dBoost': 'dBoost'
}

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


def plot_exp_scalability(results_paths, plots_path, eval_metrics, y_labels):
    """Plot scalability performance of SAGED and the baseline tools"""
    
    labels_list = []
    results_dict = {}
     
    for results_path in results_paths:
        
        if os.path.exists(results_path):

            # Read the CSV file
            results = pd.read_csv(results_path, low_memory=False)

            results_gb = results.groupby(['detector', 'split_ratio'])
            
            results_mean = results_gb[eval_metrics].mean()
            results_variance = results_gb[eval_metrics].std()

            results_keys = results_gb.groups.keys()
            
            x_values = [y for (x, y) in list(results_keys)]
            
            # parse the results in a dictionary {'saged': [x_values, means, variances]}
            results_dict[list(results_keys)[0][0]] = [x_values, results_mean, results_variance]
    
    # get a list of labels 
    labels_list = list(results_dict.keys())
    
    for plot_path, eval_metric, y_label in zip(plots_path, eval_metrics, y_labels):

        # Define list of markers
        markers_list = ['.', '+', 'x', 'o', '*', '8', 's', 'd', '1', '2', '3', '4', '|']

        fig, ax = plt.subplots()

        for idx in range(len(labels_list)):
            
            # Extract the relevant data
            label = labels_list[idx] 
            x_axis_values = [x * 200 for x in results_dict[label][0]]
            metric = list(results_dict[label][1][eval_metric])
            variance = list(results_dict[label][2][eval_metric])
            
            plt.plot(x_axis_values, metric, marker=markers_list[idx], linewidth=3, markersize=8,
                    alpha=0.6, label=labels_mapper[label])
            plt.fill_between(x_axis_values, [x - y for x,y in zip(metric, variance)], [x + y for x,y in zip(metric, variance)],
                            alpha=0.3, antialiased=True)

        plt.xlabel('Data Fraction (%)')
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
    
    # Define the list of detectors involved in the experiment 
    detectors_list = [DetectMethod.FAHES_DETECTOR.__str__(), DetectMethod.OUTLIER_DETECTOR_IF.__str__(),
                          DetectMethod.OUTLIER_DETECTOR_IQR.__str__(), DetectMethod.OUTLIER_DETECTOR_SD.__str__(),
                          DetectMethod.KATARA.__str__(), DetectMethod.DBOOST.__str__(), DetectMethod.HOLOCLEAN.__str__(),
                          DetectMethod.MIN_K.__str__(),DetectMethod.RAHA.__str__(), DetectMethod.ED2_DETECTOR.__str__(),
                          DetectMethod.SAGED.__str__()]

    for dataset_name in dataset_names:
        
        # Get the results path
        exp_name = ExperimentName.SCALABILITY.__str__()
        results_paths = []
        for method in detectors_list:
            results_paths.append(create_detections_path(EXP_PATH, 
                                                        dataset_name, 
                                                        method, 
                                                        exp_type=exp_name, 
                                                        store_detection_metrics=True,
                                                        create_new_dirs=False))


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
        plot_exp_scalability(results_paths=results_paths,
                            plots_path=plots_path,
                            eval_metrics=eval_metrics,
                            y_labels=y_labels)