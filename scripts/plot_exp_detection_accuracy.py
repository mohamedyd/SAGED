# Plot: detection accuracy (Latex table)
#   - Find the performance of SAGED and the baseline tools
# Author: Mohamed Abdelaal
# Date: May 2023

import os
import argparse
import pandas as pd
import numpy as np
from sys import exit
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

metrics_mapper = {
 'precision': 'P',
 'recall': 'R',
 'f1_score': 'F1',
 'time': 'T'   
}


def plot_exp_detection_accuracy(dataset, results_paths, eval_metrics):
    """generate a latex table of the detection accuracy of SAGED and the baseline tools"""

    # Read the CSV files
    detectors_list = []
    means_dict = {}
    variances_dict = {}
    header_tuples = []

    # Initialize a dict to store metrics where values are lists
    data = {}
    for metric in ('precision', 'recall', 'f1_score', 'time'):
        data[metrics_mapper[metric]] = []
    
    for results_path in results_paths:
        
        if os.path.exists(results_path):
            # read the CSV file
            results = pd.read_csv(results_path, low_memory=False)
            # Group the results by the detector name
            results_gb = results.groupby(['detector'])
            results_mean = results_gb[eval_metrics].mean()
        
            # Create a list of the results as a step toward creating a dataframe of the results
            for metric in ('precision', 'recall', 'f1_score', 'time'):
                data[metrics_mapper[metric]].append(round(float(results_mean[metric]), 2))
        
            # Append the labels
            detectors_list.append(labels_mapper[results['detector'][0]])
    
    
    # Create a dataframe of the results  
    results_df = pd.DataFrame(data, index=detectors_list)  
    # Create higher-level column headers
    for column in results_df.columns:
        header_tuples.append((dataset, column))
    # Add the high level header to the results dataframe
    results_df.columns = pd.MultiIndex.from_tuples(header_tuples)

    return results_df

        
if __name__ == '__main__':

    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset', nargs='+', default=None, required=True)

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset
    
    # Create an empty DataFrame to store the combined data
    df_combined = pd.DataFrame()

    for dataset_name in dataset_names:
        
        # Define the list of detectors involved in the experiment 
        detectors_list = [DetectMethod.FAHES_DETECTOR.__str__(), DetectMethod.OUTLIER_DETECTOR_IF.__str__(),
                          DetectMethod.OUTLIER_DETECTOR_IQR.__str__(), DetectMethod.OUTLIER_DETECTOR_SD.__str__(),
                          DetectMethod.KATARA.__str__(), DetectMethod.DBOOST.__str__(), DetectMethod.HOLOCLEAN.__str__(),
                          DetectMethod.MIN_K.__str__(),DetectMethod.RAHA.__str__(), DetectMethod.ED2_DETECTOR.__str__(),
                          DetectMethod.SAGED.__str__()]
        
        # Evaluation metrics
        eval_metrics = ['precision', 'recall', 'f1_score', 'time']
        # Get the results path
        exp_name = ExperimentName.DETECTION.__str__()
        results_paths = []
        for method in detectors_list:
            results_paths.append(create_detections_path(EXP_PATH, dataset_name, method, exp_type=exp_name, store_detection_metrics=True))

        # Generate a dataframe of the results for each dataset
        results_df = plot_exp_detection_accuracy(dataset=dataset_name, results_paths=results_paths, eval_metrics=eval_metrics)
        
        # Cobine the current DataFrame to the combined DataFrame
        df_combined = pd.concat([df_combined, results_df], axis=1)
        
    # Create a latex table
    latex_table = df_combined.to_latex(index=True, caption='Sample Table', column_format='lccc')
    print(latex_table)
    
    # Save the generated latex table
    table_path = os.path.join(EXP_PATH, 'evaluation', 'plots', 'detection_table.tex')
    
    with open(table_path, 'w') as f:
        f.write(latex_table)
   
    print("Table saved successfully!")