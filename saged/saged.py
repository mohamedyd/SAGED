####################################################################
# SAGED: Implement a function to execute the SAGED detector
# Authors: Mohamed Abdelaal
# Date: March 2023
# Software AG
# All Rights Reserved
####################################################################

import sys
import time
import random
import pandas as pd
import statistics
import numpy as np

from numpy import mean, where, isin
from prettytable import PrettyTable
from saged.datasets import Dataset
from saged.featurization import create_features
from saged.meta import create_meta_features, evaluate, train_meta_classifiers
from saged.labeling import active_learning, clustering_based_sampling, random_split, knn_shapley
from saged.configuration import (
    Features, ClassifierType, ClusterAlgorithm, Configuration,
    LabelingStrategy, ProfileType, Similarity)
from saged.utils import store_detections, create_detections_path, EXP_PATH, store_results_csv, DetectMethod
from baseline.setup.evaluate import evaluate_detector
from baseline.setup.utils import get_all_errors



def saged(dirty_dataset: str,
          historical_datasets: list,
          features: str = 'meta',
          profile: str = 'structure_features',
          classifier: str = 'mlp_classifier',
          clustering: str = 'kmeans',
          propagate_labels: bool = False,
          labeling: str = 'none',
          similarity: str = 'clustering',
          n_clusters: int = 1,
          n_meta_features: int = 0,
          runs: int = 1,
          num_labels: int = 20,
          label_augmentation: str = None,
          exp_type: str = 'detection',
          verbose: bool = False,
          split_ratio: float = 1.0):
    """Integrate the various steps of the SAGED detector"""
    
    # Initialize a dict to store params and metrics
    params_metrics_dict = {}
    
    # Log the input params
    params_metrics_dict.update(dirty=dirty_dataset,
                               historical=historical_datasets,
                               detector=DetectMethod.SAGED.__str__(),
                               features=features,
                               classifer=classifier,
                               clustering=clustering,
                               propagate_labels=propagate_labels,
                               labeling=labeling,
                               similarity=similarity,
                               n_cluster=n_clusters,
                               n_meta_features=n_meta_features,
                               runs=runs,
                               num_labels=num_labels,
                               label_augmentation='none' if not label_augmentation else label_augmentation,
                               exp_type=exp_type,
                               verbose=verbose,
                               split_ratio=split_ratio)

    # Load dirty and historical datasets
    dirty_dataset = Dataset(dirty_dataset)
    
    # Get configuration
    config = Configuration(
        features=Features(features.lower()),
        profile_type=ProfileType(profile.lower()),
        classifier_type=ClassifierType(classifier.lower()),
        cluster_algorithm=ClusterAlgorithm(clustering.lower()),
        labeling_strategy=LabelingStrategy(labeling.lower()),
        similarity=Similarity(similarity.lower()),
        n_clusters=n_clusters,
        n_meta_features=n_meta_features
    )

    # If --historical-datasets is not given and --features is meta, import available datasets
    if config.features is Features.META:
        if historical_datasets is None:
            historical_datasets = Dataset.load_all(skip=[dirty_dataset])
        else:
            historical_datasets = [Dataset(hd) for hd in historical_datasets]

    # Statistics
    list_f1_score = []
    list_recall = []
    list_precision = []
    list_total_time = []

    # Create a path to store the results
    results_path=create_detections_path(EXP_PATH, dirty_dataset.name, DetectMethod.SAGED.__str__(), 
                                        exp_type=exp_type, store_detection_metrics=True)

    #all_errors = get_all_errors(dirty_dataset.dirty_df, dirty_dataset.clean_df, dirty_dataset)
    
    for i in range(runs):
        if verbose:
            print(f"Iteration {i+1}")
            print("="*len(f"Iteration {i+1}"))

        start_total = time.time()

        # Create features
        if config.features is Features.META:
            if verbose:
                print("Create meta features:")

            features = create_meta_features(dirty_dataset, historical_datasets, config, verbose=verbose)
        else:
            if verbose:
                print("Create classic features...")

            classic_features = create_features(dirty_dataset.dirty_df)
            features = pd.concat(classic_features.values(), axis="columns", keys=classic_features.keys())

        if verbose:
            print("Features created.\n")

        
        if verbose:
            print(f"Label samples (strategy: {config.labeling_strategy.value}):")

        if config.labeling_strategy is LabelingStrategy.NONE:
            X_train, X_test, y_train, y_test, train_indices, test_indices = random_split(
                features, dirty_dataset.get_actual_errors(), num_labels=num_labels)
            print(dirty_dataset.get_actual_errors().head(30))
            
        elif config.labeling_strategy is LabelingStrategy.HEURISTIC:
            # calculate the number of ones in each row
            num_ones = features.sum(axis=1)
            # retrieve the indexes of the rows with a large number of ones
            train_indices = num_ones.nsmallest(n=num_labels).index
            # extract X_train and X_test
            X_train = features.loc[train_indices]
            X_test = features.drop(train_indices)
            # get the index of the test data
            test_indices = list(X_test.index)
            
            # extract the labels
            all_errors = dirty_dataset.get_actual_errors()
            y_train = all_errors.loc[train_indices]
            y_test = all_errors.drop(train_indices)
            
        elif config.labeling_strategy is LabelingStrategy.CLUSTERING:
            X_train, X_test, y_train, y_test, train_indices, test_indices = clustering_based_sampling(
                features, dirty_dataset.get_actual_errors(), propagate=propagate_labels,
                num_labels=num_labels, verbose=verbose)
            
        elif config.labeling_strategy is LabelingStrategy.ACTIVE_LEARNING:
            X_train, X_test, y_train, y_test = active_learning(
                features, dirty_dataset.get_actual_errors(), config.classifier_type,
                num_labels=num_labels, verbose=verbose)
            train_indices = X_train.index
            test_indices = X_test.index

        if verbose:
            print("Samples labeled.\n")

        if verbose:
            print("Train classifiers:")

        meta_classifiers = train_meta_classifiers(X_train, y_train, config.classifier_type, verbose=verbose)
        
        if verbose:
            print("Classifiers trained.\n")

        precision, recall, f1_score, detection_dict, y_pred = evaluate(meta_classifiers, 
                                                                       X_test, 
                                                                       y_test, 
                                                                       y_train, 
                                                                       train_indices, 
                                                                       test_indices)
        if verbose:
            print(f"Length of the initial detection dictionary (no label augmentation): {len(detection_dict)}")
        

        """
        ============================== Labels Augmentation ============================================ 
        Use the obtained predictions to enahnce the training data, before retraining the meta models
        """   
            
        if label_augmentation != None:
            
            if label_augmentation == "random":
                # Augment labels where the tuples selected randomly
                # Split the test data into two parts. Use 20% of the test data to augment the training data
                X_train2, _, y_train2, _, _, _ = random_split(X_test, y_pred, num_labels=0.1, random_seed=random.randint(0, 100))
            
            elif label_augmentation == "prediction":
                # Extract indexes of the rows containing dirty cells
                selected_rows = [id[0] for id in detection_dict.keys()]
                # remove duplicates
                selected_rows_no_duplicates = list(dict.fromkeys(selected_rows))
                # removing the indexes of the user-labeled training data (since they do not exist in y-pred)
                selected_rows_filtered = [element for element in selected_rows_no_duplicates if element not in train_indices]
                # Create second training set to retrain the detection meta classifiers 
                X_train2 = features.loc[selected_rows_filtered,:]
                # Create second set of training labels extracted from the predictions of the initial meta classifiers 
                y_train2 = y_pred.loc[selected_rows_filtered,:]

            # Use active learning to select the most important tuples
            elif label_augmentation == "active_learning":
                X_train2, _, y_train2, _ = active_learning(X_test, y_pred, config.classifier_type, num_labels=0.2, verbose=verbose)

            # Execute active data acquisition using KNN-Shap
            elif label_augmentation == "knn_shapley": 
                
                # Split the test data into two parts. Use 20% of the test data to augment the training data
                X_train_shap, X_test_shap, y_train_shap, y_test_shap, _, _ = random_split(X_test, y_pred, num_labels=0.2)
                    
                if verbose:
                    print("Finding the tuple importance of the meta features")
                    
                X_train2, y_train2 = knn_shapley(X_train_shap, X_test_shap, y_train_shap, y_test_shap)
            
            else:
                raise NotImplementedError
            
            # Augment the training data with the obtained predictions (from X_test and y_pred)
            augmented_X_train = pd.concat([X_train2, X_train])
            augmented_y_train = pd.concat([y_train2, y_train])
            
            if verbose:
                print("Size of X_train: {}".format(X_train.shape))
                print("Size of y_train: {}".format(y_train.shape))
                print("Size of augmented X_train: {}".format(augmented_X_train.shape))
                print("Size of augmented y_train: {}".format(augmented_y_train.shape))
                print("Retrain classifiers:")

            # Use the augmented training data to retrain the meta classifiers            
            meta_classifiers2 = train_meta_classifiers(augmented_X_train, 
                                                       augmented_y_train, 
                                                       config.classifier_type, 
                                                       verbose=verbose)
            if verbose:
                print("Classifiers retrained.")
            
            precision, recall, f1_score, detection_dict, y_pred = evaluate(meta_classifiers2,
                                                                              X_test, 
                                                                              y_test, 
                                                                              y_train, 
                                                                              train_indices, 
                                                                              test_indices) 
            if verbose:
                # Initial and final dictionaries will be the same if ADA is not used or if it makes no difference
                print(f"Length of the detection dictionary (with label augmentation): {len(detection_dict)}") 
        
        # Estimate the detection time of SAGED
        end_total = time.time()
        time_total = end_total - start_total
        
        # Log the detection accuracy metrics
        params_metrics_dict.update(time=time_total,
                                   precision=precision,
                                   recall=recall,
                                   f1_score=f1_score)
        
        # Evaluate detections relative to the actual errors in the data, but first find all errors in the dataset
        #all_errors = get_all_errors(dirty_dataset.dirty_df, dirty_dataset.clean_df, dirty_dataset)
        #precision2, recall2, f1_score2 = evaluate_detector(all_errors=all_errors, detections=detection_dict)   
        #print(f"using comparison to actual errors: precision:{precision2}, recall:{recall2}, f1:{f1_score2}")
        
        if verbose and runs > 1:
            print(f"* Precision: {precision}, recall: {recall}, F1 score: {f1_score}, ",
                f"total time: {time_total}\n")

        # Store the results
        store_results_csv(params_metrics_dict, results_path)

        list_precision.append(precision)
        list_recall.append(recall)
        list_f1_score.append(f1_score)
        list_total_time.append(time_total)

    print(f"\nDirty dataset: {dirty_dataset}")
    if config.features is Features.META:
        print("Historical datasets:\n * {}".format('\n * '.join([str(hd) for hd in historical_datasets])))
        print("Total amount of elements in historical datasets: ", end="")
        print(sum(hd.dirty_df.size for hd in historical_datasets))
        
    table = PrettyTable([""] + list(range(1, runs+1)) + ["Mean values"])
    table.add_row(["Precision"] + list_precision + [mean(list_precision)])
    table.add_row(["Recall"] + list_recall + [mean(list_recall)])
    table.add_row(["F1 score"] + list_f1_score + [mean(list_f1_score)])
    table.add_row(["Total time"] + list_total_time + [mean(list_total_time)])
    table.align = "r"
    table.float_format = ".2"
    print(table)

    if runs > 1:
        print("Precision standard deviation: ", statistics.stdev(list_precision))
        print("Recall standard deviation: ", statistics.stdev(list_recall))
        print("F1-Score standard deviation: ", statistics.stdev(list_f1_score))
        print("Total Time standard deviation: ",  statistics.stdev(list_total_time))
        
     
    # Store detections in detector directory
    detections_path = create_detections_path(EXP_PATH, dirty_dataset.name, DetectMethod.SAGED.__str__(), exp_type=exp_type) 
    store_detections(detection_dict, detections_path)

    return list_precision, list_recall, list_f1_score, list_total_time