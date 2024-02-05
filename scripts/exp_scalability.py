# Scalability Analysis 
#   - Compare the scalability of SAGED with respect to a number of baseline tools
#   - Baselines: raha, ed2, SD, IQR, IF, min-k, fahes, holoclean, dboost, katara
#
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
from baseline.model.utils import ExperimentName, ExperimentType, create_results_path
from scripts.plot_exp_scalability import plot_exp_scalability


def exp_scalability(datasets, historical_datasets, detectors_list, exp_type, runs, verbose):
    """
    The method uses stratified sampling to generate #of splits of the GT and dirty data before running
        the scalability analysis for the error detection and repair methods
    datasets -- list of strings denoting the examined datasets
    settings -- dictionary of all configurations
    :return:
    """

    for dataset in datasets:
        
        # Load dirty and historical datasets
        dataset_object = Dataset(dataset)
        dirtyDF = dataset_object.dirty_df
        groundtruthDF = dataset_object.clean_df

        # Rename the clean and dirty CSV files to generate new splits
        new_clean_path = os.path.abspath(os.path.join(dataset_object.directory,'clean_temp.csv'))
        new_dirty_path = os.path.abspath(os.path.join(dataset_object.directory, 'dirty_temp.csv'))
        os.rename(os.path.join(dataset_object.directory, 'clean.csv'), new_clean_path)
        os.rename(os.path.join(dataset_object.directory, 'dirty.csv'), new_dirty_path)

        # Get the labels
        labels = dataset_object.cfg.labels

        for percent in np.arange(0.1, 1.1, 0.1):

            # Handle the case of percent = 1
            if percent == 1:
                # Restore the original datasets
                os.rename(new_clean_path, os.path.join(dataset_object.directory, 'clean.csv'))
                os.rename(new_dirty_path, os.path.join(dataset_object.directory, 'dirty.csv'))
                # TODO add here SAGED and other detectors
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
                
                break
            else:
                # Initalize a stratified sampler
                split = StratifiedShuffleSplit(n_splits=1, test_size=percent, random_state=42)

                # ==== Generate data splits ===
                # Sample the GT and dirty data according to the generated indices
                for train_index, test_index in split.split(groundtruthDF, groundtruthDF[labels]):
                    test_gt_set = groundtruthDF.loc[test_index]
                    test_dirty_set = dirtyDF.loc[test_index]

                # Set the path to save the splits
                if verbose:
                    print('Generating new data splits with percent: {}'.format(percent))

                out_gt_path = os.path.abspath(os.path.join(dataset_object.directory, 'clean.csv'))
                out_dirty_path = os.path.abspath(os.path.join(dataset_object.directory, 'dirty.csv'))
                test_gt_set.to_csv(out_gt_path, sep=",", index=False, encoding="utf-8")
                test_dirty_set.to_csv(out_dirty_path, sep=",", index=False, encoding="utf-8")

                try:
                    # TODO add here SAGED and other detectors
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
                
    
                except Exception as e:
                    logging.info("Exception: {}".format(e.args[0]))
                    logging.info("Exception: {}".format(sys.exc_info()))
                    continue

                # Remove the current split and its relevant files
                os.remove(out_gt_path)
                os.remove(out_dirty_path)
                # Remove detection files
                raha_files_path = os.path.abspath(os.path.join(dataset_object.directory, 'raha-baran-results-{}'.format(dataset)))
                shutil.rmtree(raha_files_path, ignore_errors=True)

        #os.rename(new_clean_path, os.path.join(dataset_object.directory, 'clean.csv'))
        #os.rename(new_dirty_path, os.path.join(dataset_object.directory, 'dirty.csv'))

        # =================== Plot the results ==================================
        
        # Get the results path
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
        plot_exp_scalability(results_paths=results_paths,
                            plots_path=plots_path,
                            eval_metrics=eval_metrics,
                            y_labels=y_labels)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

    parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")
    
    args = parser.parse_args()

    # Define the parameters of the experiment 
    dirty_datasets = ['flights', 'tax', 'soccer', 'smartfactory']
    
    historical_datasets = ['adult', 'movies_1']
    involved_detectors = [DetectMethod.ED2_DETECTOR, DetectMethod.RAHA,
                        DetectMethod.KATARA, DetectMethod.DBOOST,
                        DetectMethod.FAHES_DETECTOR, DetectMethod.HOLOCLEAN,
                        DetectMethod.MIN_K, DetectMethod.OUTLIER_DETECTOR_SD,
                        DetectMethod.OUTLIER_DETECTOR_IQR, DetectMethod.OUTLIER_DETECTOR_IF]

    involved_detectors = [DetectMethod.OUTLIER_DETECTOR_IF, DetectMethod.KATARA]

    exp_type = ExperimentName.SCALABILITY.__str__()

    exp_scalability(dirty_datasets, historical_datasets, involved_detectors, exp_type, args.runs, args.verbose)