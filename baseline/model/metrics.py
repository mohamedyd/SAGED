#####################################################################################
# Metrics: Implement a set of evaluation metrics to assess the predictive performance
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
####################################################################################

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import RepeatedKFold, cross_val_score


def evaluate_model(metrics_dict, labels, predictions, ml_task, verbose=True):
    """Evaluate the predictive accuracy """
    # Check that both predictions and labels are of equal length
    if len(labels) == len(predictions):
        if ml_task == 'regression':
            mse = mean_squared_error(labels, predictions)
            r2 = r2_score(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            metrics_dict.update(mse=mse, r2=r2, mae=mae)
            if verbose:
                print("Evaluating {} model".format(ml_task))
                print(
                    f"MSE: {mse}\n"
                    f"R2: {r2}\n"
                    f"MAE: {mae}"
                )
            return metrics_dict
        elif ml_task in ['binary_classification', 'multiclass_classification']:
            average = 'micro' if ml_task == 'multiclass_classification' else 'binary'
            precision = precision_score(labels, predictions, average=average)
            recall = recall_score(labels, predictions, average=average)
            f1 = f1_score(labels, predictions, average=average)
            metrics_dict.update(precision=precision, recall=recall, f1_score=f1)
            if verbose:
                print("Evaluating {} model".format(ml_task))
                print(
                    f"Precision: {precision}\n"
                    f"Recall: {recall}\n"
                    f"F1 Score: {f1}"
                )
            return metrics_dict
        else:
            raise NotImplementedError
    else:
        raise ValueError("Predictions and labels are of different lengths!")


def cross_validation(X, y, model, verbose=True):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    results = np.absolute(scores)
    if verbose:
        print('Mean MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))
