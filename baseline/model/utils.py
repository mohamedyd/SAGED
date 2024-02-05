#######################################################################
# Utils: Implement several helping functions used while training models
# regression
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
#######################################################################

import os
import csv
import pandas as pd
import time
from enum import Enum
from baseline.setup.detectors.detect_method import EXP_PATH
import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib

mpl.rc('axes', labelsize=16)
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 16})

lc_mapper = {'mean_squared_error': 'Training MSE',
             'val_mean_squared_error': 'Validation MSE',
             'binary_accuracy': 'Training Accuracy',
             'val_binary_accuracy': 'Validation Accuracy'}

class ExperimentType(Enum):
    AUG2CLEAN = 'baseline'
    GROUND_TRUTH = 'clean'
    E2E_PIPELINE = 'e2e'

    def __str__(self):
        return self.value


class ExperimentName(Enum):
    LABELING_BUDGET= 'labeling_budget'
    LABELS_AUGMENTATION= 'labels_augmentation'
    LABELING_STRATEGY= 'labeling_strategy'
    ERROR_RATE = 'error_rate'
    SCALABILITY= 'scalability'
    RUNTIME = 'runtime'
    SIMILARITY= 'similarity'
    PREDICTIVE_ACCURACY = 'predictive-accuracy'
    DETECTION = 'detection'
    REPAIR = 'repair'
    MODELING = 'e2e'
    OUTLIER_DEGREE='outlier_degree'


    def __str__(self):
        return self.value


class EvaluationDir(Enum):
    PLOTS = 'plots'
    DATA = 'data'

    def __str__(self):
        return self.value


def save_fig(fig_path, tight_layout=True, fig_extension="png", resolution=300, verbose=True):
    if verbose:
        print("[INFO] Saving the learning curve ..")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)


def create_results_path(dataset_name,
                        experiment_name,
                        filename,
                        exp_path=EXP_PATH,
                        parent_dir='data',
                        return_results_dir=False):
    """
    Creates a path to the results file
    """
    # Create a directory for such experiments, called repair
    eval_model_path = os.path.abspath(os.path.join(exp_path, 'evaluation', parent_dir, dataset_name, experiment_name))
    if not os.path.exists(eval_model_path):
        pathlib.Path(eval_model_path).mkdir(parents=True)

    target_path = os.path.abspath(os.path.join(eval_model_path, filename))
    if return_results_dir:
        return target_path, eval_model_path
    else:
        return target_path


def store_results_csv(results, target_path):
    """Store the results in CSV files"""

    # Check whether the file already exists: useful for writing the header only once
    file_exists = os.path.isfile(target_path)

    # Store the results
    with open(target_path, 'a', newline='') as f_object:
        w = csv.DictWriter(f_object, results.keys())

        if not file_exists:
            # Write the header
            w.writeheader()

        # write the results
        w.writerow(results)

        # Close the file object
        f_object.close()


def plot_learning_curves(history,
                         dataset_name,
                         experiment_type=ExperimentType.AUG2CLEAN.__str__(),
                         metric_name='Accuracy',
                         fig_extension='pdf'):
    """
    Plot learning curves for Scikeras regressors and classifiers
    """
    history_df = pd.DataFrame(history)
    history_df = history_df.drop(['loss', 'val_loss'], axis=1)
    if 'accuracy' in history_df:
        history_df = history_df.drop(['accuracy', 'val_accuracy'], axis=1)

    # Renaming the labels
    for column in history_df.columns:
        history_df.rename(columns={column: lc_mapper[column]}, inplace=True)

    # history_df.plot(figsize=(8, 5))
    history_df.plot(linewidth=3,fontsize=16)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    # plt.gca().set_ylim(0, 1)

    # Prepare the figure path
    timestr = time.strftime("%Y%m%d_%H%M%S")
    fig_id = "lc" + "_" + experiment_type + "_" + dataset_name + "_" + timestr + "." + fig_extension
    fig_path = create_results_path(dataset_name=dataset_name,
                                   experiment_name=ExperimentName.LEARNING_CURVE.__str__(),
                                   parent_dir=EvaluationDir.PLOTS.__str__(),
                                   filename=fig_id)
    save_fig(fig_path, fig_extension=fig_extension)
