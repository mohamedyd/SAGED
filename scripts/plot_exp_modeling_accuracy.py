# Plot: modeling accuracy (spider plot)
#   - Compare the performance of E2E-SAGED pipline and the baseline E2E piplines

# Author: Mohamed Abdelaal
# Date: May 2023

import matplotlib
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from matplotlib.path import Path
from adjustText import adjust_text
import matplotlib.patches as mpatches
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.cm import tab10
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from baseline.dataset.dataset import Dataset

import os
import argparse
import pandas as pd
import numpy as np
from sys import exit
from saged.utils import create_detections_path, EXP_PATH
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.model.utils import create_results_path, ExperimentName, ExperimentType

# Mapping dictionaries
detectors_mapper = {'min_k': ['Min', 'M'],
                    'min': ['Min', 'M'],
                    'metadata_driven': ['Meta', 'T'],
                    'ed2': ['ED2', 'E'],
                    'raha': ['RAHA', 'R'],
                    'katara': ['Katara', 'K'],
                    'mvdetector': ['MVD', 'V'],
                    'max_entropy': ['Max', 'X'],
                    'holoclean': ['Holo', 'H'],
                    'nadeef': ['NADEEF', 'N'],
                    'openrefine': ['OpnR', 'O'],
                    'picket': ['Picket', 'P'],
                    'fahes_ALL': ['FAHES', 'F'],
                    'fahes': ['FAHES', 'F'],
                    'duplicatesdetector': ['DuplD', 'D'],
                    'dboost': ['dBoost', 'B'],
                    'dBoost': ['dBoost', 'B'],
                    'IF': ['IF', 'I'],
                    'SD': ['SD', 'S'],
                    'IQR': ['IQR', 'Q'],
                    'zeroer': ['ZeroER', 'Z'],
                    'cleanlab': ['Cleanlab', 'C'],
                    'aug2clean': ['AutoCure', 'A'],
                    'cleanlab-forest_clf': ['Cleanlab', 'C'],
                    'dirty': ['', 'R'],
                    'saged': ['SAGED', 'SG']}
cleaners_mapper = {'cleanWithGroundTruth': ['GT', '1'],
                   'dirty': ['Dirty', ''],
                   'baran': ['BARAN', '4'],
                   'dcHoloCleaner-without_init': ['Holo', '5'],
                   'mlImputer': ['ML-Impute', '2'],
                   'standardImputer': ['Impute', '3'],
                   }

cleaners_list = cleaners_mapper.keys()
detectors_list = detectors_mapper.keys()


# ========== Plot settings =======
plt.rc('grid', linestyle=":", color='black', alpha=0.3)
bar_width = 0.3
bar_width_runtime = 0.9
opacity = 0.4
error_config = {'ecolor': '0.3'}
matplotlib.rcParams.update({'font.size': 16})
colors = ["g", "b", "r", "y", "c", "m", "k", "orange"]
colors2 = ["red", "green"]
patterns = ["/", "o", "\\", "x", "|", ".", "*", "-", "+", "O", '/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']

# ===========================================
#            Helper Functions
# ===========================================

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def results_parser(results_directory, data_name, evaluation_metric, taking_log=False):
    """
    This method parses the results of detection, repairing, and modeling to automatically plot the figures
    :param file_df: dataframe of the obtained resutls in each task
    :param task: string denoting whether detection, repair, or ml modeling
    :return: array containing the paramters to be plotted
    """

    # Initialize a list to store the names of the detector-repair combinations
    combined_names = []
    # Initialize a list to store the results of clean data, aug2clean, and other baselines
    results_list = []
    clean_results = []

    # Get the list of CSV files in the data path
    for _, _, files in os.walk(results_directory):

        # Create paths to all files in the results directory
        for name in files:
            if name == 'clean_' + data_name + '.csv':
                clean_results = pd.read_csv(os.path.join(results_directory, name))[evaluation_metric].tolist()
            elif name == 'saged_' + data_name + '.csv':
                aug2clean_results = pd.read_csv(os.path.join(results_directory, name))[evaluation_metric]
                combined_names.append(detectors_mapper['aug2clean'][0])
                results_list.append(statistics.mean(aug2clean_results))


            elif 'e2e' in name:
                detector_name = detectors_mapper[name.split('_')[1]][0]
                repair_name = cleaners_mapper[name.split('_')[2]][0]
                combined_names.append(detector_name + '-' + repair_name)
                e2e_results = pd.read_csv(os.path.join(results_directory, name))[evaluation_metric]
                results_list.append(statistics.mean(e2e_results))
            else:
                raise NotImplemented

            if taking_log:
                clean_results = np.log(clean_results)
                results_list = np.log(results_list)

        return combined_names, clean_results, results_list



def plot_radar(results_directory, evaluation_metric, dataset_name, plot_path):
    """
    Plot the radar figure for comparing aug2clean and other baseline methods in terms of preditctive accuracy
    """
    # Load the relevant data
    combin_names, clean_results, compared_results = results_parser(results_directory=results_directory,
                                                                   data_name=dataset_name,
                                                                   evaluation_metric=evaluation_metric)
    
    # Rename the inputs to be used in the plotting function
    s4_sel = clean_results
    s1_sel = compared_results

    # Adjust the values in S4
    adjusted_values = []
    mean = np.mean(s4_sel)
    for _ in np.arange(len(s1_sel)):
        adjusted_values.append(mean)
    s4_sel = adjusted_values
    
    # ========= Map the strategy names to indices, e.g., KAT-BARAN --> B15
    indices = []
    for name in combin_names:
        # Handle the case of dirty version of the dataset
        if name == 'Dirty':
            indices.append('D0')
        else:
            # Each name has the form det-cln, e.g., RAHA-MISS-MIX
            # Extract the detector
            det = name.split('-')[0]
            # Extract the cleaner using the detector, +1 is used to count for the dash
            cln = name[len(det) + 1:]

            # Initialize the index of the detector
            first_index = ''
            # Initialize the index of the cleaner
            second_index = ''
            # Get the detector index using the detector abbreviation
            for raw_det in detectors_list:
                if det == detectors_mapper[raw_det][0]:
                    first_index = detectors_mapper[raw_det][1]
                    break
            # Get the cleaner index using the cleaner abbreviation
            for raw_cln in cleaners_list:
                if cln == cleaners_mapper[raw_cln][0]:
                    second_index = cleaners_mapper[raw_cln][1]
                    break
            # Concatenate the two indices to form the index of the strategy and append to the list
            indices.append(first_index + second_index)
            
    # ============ Map the strategy names to numerical values
    combin = np.arange(len(combin_names)) + 1
    
    # move D0 to the front of the list to make sure it appears in the plot
    if 'SG1' in indices:
        d0_index = indices.index('SG1')
        d0_result = s1_sel[d0_index]
        # remove D0 and its result
        indices.remove('SG1')
        #s1_sel.iloc[d0_index]
        s1_sel_list = list(s1_sel) # Convert Series into list
        s1_sel_list.remove(d0_result)

        indices.insert(0, 'SG1') # this is not enough, we shall also modify the list of results
        s1_sel_list.insert(0, d0_result)
        s1_sel = pd.Series(s1_sel_list)
           
           
    # ============= Reduce the number of labels if they excedded 40
    if len(indices) > 40:
        new_combin = []
        new_indices = []
        for item in np.arange(len(indices)):
            # Set a reduction factor, e.g., if 3 is used then only 3,6,9,.. are used
            reduction_factor = 3
            if item == len(indices) - 1:
                # Avoid overlap between first and last label
                new_indices.append('')
            else:
                #new_combin.append(str(item)) if item % reduction_factor == 0 else new_combin.append('')
                new_indices.append(indices[item]) if item % reduction_factor == 0 else new_indices.append('')

        #combin = new_combin
        indices = new_indices

    # ============== Estimate the angles
    theta = radar_factory(len(indices), frame='polygon')

    # =============== Instantiate a figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1])
    # For loop for controlling the style of the marker per point
    for i, n, m in zip(range(len(theta)), theta, s4_sel):
        if i == 0:
            ax.plot(n, m, color='blue', marker='s', label='Repaired')
        else:
            # Avoid adding the label to the legend several times
            ax.plot(n, m, color='blue', marker='s')
    ax.fill(theta, s1_sel, facecolor='blue', alpha=0.25)
    ax.plot(theta, s4_sel, color='g', marker='.', label='Ground Truth')
    ax.fill(theta, s4_sel, facecolor='g', alpha=0.25)

    # Draw one axe per variable + add labels
    plt.xticks(theta, indices)

    # Add legend
    plt.legend(loc='best')

    figure_name = 'radar_{}.pdf'.format(dataset_name)
    figure_path = os.path.join(plot_path, figure_name)
    plt.savefig(figure_path, dpi=1200, orientation='portrait', format='pdf', bbox_inches='tight')
    
    

if __name__ == '__main__':

    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset-name', nargs='+', default=None, required=True)

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_names = args.dataset_name

    for dataset_name in dataset_names:
        # Create a data object
        data_obj = Dataset(dataset_name)
        # get the evaluation metric to be samples from the CSV file
        eval_metric = 'mse' if data_obj.cfg.ml_task == 'regression' else 'f1_score'

        exp_name = ExperimentName.MODELING.__str__()
        filename = ''
        _, results_dir = create_results_path(dataset_name=dataset_name,
                                             experiment_name=exp_name,
                                             filename=filename,
                                             return_results_dir=True)
        plot_path = create_results_path(dataset_name=dataset_name,
                                             experiment_name=exp_name,
                                             filename=filename,
                                             parent_dir='plots')

        plot_radar(results_directory=results_dir,
                   dataset_name=dataset_name,
                   evaluation_metric=eval_metric,   # evaluation_metric='f1_score',
                   plot_path=plot_path)