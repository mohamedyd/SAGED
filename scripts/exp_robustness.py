# Robustness analysis
#   - Evaluate the ability of SAGED and baseline methods to detect outliers with different severity
# Author: Mohamed Abdelaal
# Date: May 2023

import os
import argparse
import pandas as pd
import numpy as np
from sys import exit
import shutil
from baseline.dataset.dataset import Dataset
from saged.saged import saged
from sklearn.model_selection import StratifiedShuffleSplit
from saged.utils import create_detections_path, EXP_PATH
from scripts.run_baseline import run_baselines
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.setup.create_dirty import inject_errors, ErrorType
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_robustness import plot_exp_robustness


def exp_robustness(datasets, historical_datasets, detectors_list, evaluate_error_rate, exp_type, runs, verbose):
    """
        The method examines performance of error detection and repair for different error rates and different outlier degrees
        datasets -- list of strings denoting the examined datasets
        :return:
    """

    for dataset in datasets:
        
        # Load dirty and historical datasets
        dataset_object = Dataset(dataset)
        groundtruthDF = dataset_object.clean_df
        
        # Get the labels
        labels = dataset_object.cfg.labels
        
        # Rename the dirty CSV files to generate new dirty versions
        new_dirty_path = os.path.abspath(os.path.join(dataset_object.directory, 'dirty_temp.csv'))
        os.rename(os.path.join(dataset_object.directory, 'dirty.csv'), new_dirty_path)

        # Define the range of error rates or outlier degrees
        iter_range = np.arange(0.1, 1.1, 0.1) if evaluate_error_rate else np.arange(1, 11, 1)

        for percent in iter_range:
            # ==== Generate dirty version ===
            # Set the configurations
            configurations = [percent, 3] if evaluate_error_rate else [0.3, percent]
            
            # Inject outliers
            if verbose:
                print("Generating a dirty version with percent: {}".format(percent))
            
            if not evaluate_error_rate:
                inject_errors(clean_df=groundtruthDF,
                              error_types=[ErrorType.OUTLIERS], 
                              params=configurations, 
                              muted_columns=labels,
                              data_path=dataset_object.directory)
            else:
                inject_errors(clean_df=groundtruthDF, 
                              error_types=[ErrorType.OUTLIERS, [ErrorType.EXPLICIT_MV.func]], 
                              params=configurations, 
                              muted_columns=[labels],
                              data_path=dataset_object.directory)

            # Run the detectors on the generated dirty data
            list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=dataset,
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
                                            runs=runs,
                                            num_labels=20,
                                            label_augmentation=None,
                                            verbose=verbose,
                                            exp_type=exp_type,
                                            split_ratio=percent
                                            )
            # Run the baseline methods
            run_baselines(involved_detectors=detectors_list, 
                                dataset_names=[dataset], 
                                num_labels=20, 
                                nb_runs=runs, 
                                experiment_type=exp_type, 
                                verbose=verbose,
                                split_ratio=percent)
            
            # Remove the current split and its relevant files
            os.remove(os.path.join(dataset_object.directory, 'dirty.csv'))
            
            # Remove detection files
            raha_files_path = os.path.abspath(os.path.join(dataset_object.directory, 'raha-baran-results-{}'.format(dataset)))
            shutil.rmtree(raha_files_path, ignore_errors=True)
            
        # Restore the original dirty version
        os.rename(new_dirty_path, os.path.join(dataset_object.directory, 'dirty.csv'))

    # =================== Plot the results ==================================
    
        results_paths = []
        detectors_list.append(DetectMethod.SAGED)
        for method in detectors_list:
            results_paths.append(create_detections_path(EXP_PATH, 
                                                        dataset, 
                                                        method.__str__(), 
                                                        exp_type=exp_type, 
                                                        store_detection_metrics=True,
                                                        create_new_dirs=False))


        plots_path = []
        # Get a path to the accuracy figure
        fig_name = 'accuracy' + '_' + dataset + '.pdf'
        plots_path.append(create_results_path(dataset_name=dataset,
                                             experiment_name=exp_type,
                                             filename=fig_name,
                                             parent_dir='plots'))

        # Get a path to the execution time figure
        fig_name = 'time' + '_' + dataset + '.pdf'
        plots_path.append(create_results_path(dataset_name=dataset,
                                             experiment_name=exp_type,
                                             filename=fig_name,
                                             parent_dir='plots'))


        # Plot the results of Aug2Clean and the clean dataset
        y_labels = ['F1 Score', 'Detection time (S)']
        eval_metrics = ['f1_score', 'time']
        plot_exp_robustness(results_paths=results_paths,
                            plots_path=plots_path,
                            eval_metrics=eval_metrics,
                            y_labels=y_labels,
                            plot_outlier_degree= not evaluate_error_rate)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

    parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")
    parser.add_argument('--evaluate-error-rate', action="store_true", default=False, help="execute error rate experiment")

    
    args = parser.parse_args()

    # Define the parameters of the experiment 
    evaluate_error_rate = args.evaluate_error_rate
    
    dirty_datasets = ['nasa', 'beers', 'bikes', 'smartfactory', 'flights', 'restaurants', 'hospital']
    
    historical_datasets = ['adult', 'movies_1']
    involved_detectors = [DetectMethod.ED2_DETECTOR, DetectMethod.RAHA,
                        DetectMethod.KATARA, DetectMethod.DBOOST,
                        DetectMethod.FAHES_DETECTOR, DetectMethod.HOLOCLEAN,
                        DetectMethod.MIN_K, DetectMethod.OUTLIER_DETECTOR_SD,
                        DetectMethod.OUTLIER_DETECTOR_IQR, DetectMethod.OUTLIER_DETECTOR_IF]


    if evaluate_error_rate:
        exp_type = ExperimentName.ERROR_RATE.__str__()
    else:
        exp_type = ExperimentName.OUTLIER_DEGREE.__str__()

    exp_robustness(datasets=dirty_datasets, historical_datasets=historical_datasets, 
                   detectors_list=involved_detectors, evaluate_error_rate=evaluate_error_rate,
                   exp_type=exp_type, runs=args.runs, verbose=args.verbose)