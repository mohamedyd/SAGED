###################################################################
# ActiveClean: implement activeClean to jointly clean data and train models

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import logging
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from baseline.dataset.preprocess import preprocess
from baseline.dataset.dataset import Dataset
from scipy.sparse import vstack


def activeClean(dirty_df, clean_df, dataset_name, detections, sampling_budget):
    """
    Runs activeclean, which does not clean the data but uses a specific sampling strategy to samples data entries that
    should be cleaned by an human annotator. The human annotator is simulated by the ground truth dataset. The
    method uses gradient descent strategy to sample the optimal data entries. It therefore needs a maximum number of
    samples which are examined.
    Sampling_budget * number_of_entries sets this number.
    Note: ActiveClean works only with datasets associated with binary classification tasks

    @arguments:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    configs: sampling_budget (float) -- sampling budget in percent of the total number of entries. e.g. sampling budget = 0.2 means that
                                        20% of all data are used.

    Returns:
    results (dict) -- dictionary with p, r, f1 for the applicable scenarios
    app -- instance of model class that can be used to store results
    """

    def activeclean_orig(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=1000):
        # makes sure the initialization uses the training data
        X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
        y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

        X_clean = clean_data[0]
        y_clean = clean_data[1]

        X_test = test_data[0]
        y_test = test_data[1]

        logging.info("[ActiveClean Real] Initialization")

        lset = set(indextuple[2])

        dirtyex = [i for i in indextuple[0]]
        cleanex = []

        total_labels = []

        if len(indextuple[1]) == 0:
            raise ValueError("no dirty tuples to be cleaned")

        import time
        timeout = time.time() + 5 * 60  # now + 2 minutes

        # Initialize examples_real
        examples_real = []

        while len(np.unique(y_clean[cleanex])) < len(np.unique(y_clean)):

            # Not in the paper but this initialization seems to work better, do a smarter initialization than
            # just random sampling (use random initialization)
            topbatch = np.random.choice(range(0, len(dirtyex)), batchsize, replace=False)
            examples_real = [dirtyex[j] for j in topbatch]
            examples_map = translate_indices(examples_real, indextuple[2])
            cleanex.extend(examples_map)
            if time.time() > timeout:
                raise TimeoutError("Could not find warm up configuration with clean data")

        for j in examples_real:
            dirtyex.remove(j)

        clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
        clf.fit(X_clean[cleanex, :], y_clean[cleanex])

        for i in range(20, total, batchsize):

            logging.info("[ActiveClean Real] Number Cleaned So Far %s", len(cleanex))
            ypred = clf.predict(X_test)
            logging.info("[ActiveClean Real] Prediction Freqs %s, %s", np.sum(ypred), np.shape(ypred))
            logging.info(classification_report(y_test, ypred))

            # Sample a new batch of data
            examples_real = np.random.choice(dirtyex, batchsize)
            examples_map = translate_indices(examples_real, indextuple[2])

            total_labels.extend([(r, (r in lset)) for r in examples_real])

            # on prev. cleaned data train error classifier
            ec = error_classifier(total_labels, full_data)

            for j in examples_real:
                try:
                    dirtyex.remove(j)
                except ValueError:
                    pass

            dirtyex = ec_filter(dirtyex, full_data, ec)

            # Add Clean Data to The Dataset
            cleanex.extend(examples_map)

            # uses partial fit (not in the paper--not exactly SGD)
            clf.partial_fit(X_clean[cleanex, :], y_clean[cleanex], classes=np.unique(y_clean))

            logging.info("[ActiveClean Real] Accuracy %s, %s", i, accuracy_score(y_test, ypred))

            if len(dirtyex) < batchsize:
                logging.info("[ActiveClean Real] No More Dirty Data Detected")
                break

        average = "micro" if len(np.unique(y_test)) > 2 else "binary"

        p, r, f1, _ = precision_recall_fscore_support(y_test, ypred, average=average)
        return p, r, f1

    def translate_indices(globali, imap):
        lset = set(globali)
        return [s for s, t in enumerate(imap) if t in lset]

    def error_classifier(total_labels, full_data):
        indices = [i[0] for i in total_labels]
        labels = [int(i[1]) for i in total_labels]
        if np.sum(labels) < len(labels):
            clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
            clf.fit(full_data[indices, :], labels)

            return clf
        else:
            return None

    def ec_filter(dirtyex, full_data, clf, t=0.90):
        if clf != None:
            pred = clf.predict_proba(full_data[dirtyex, :])

            return [j for i, j in enumerate(dirtyex) if pred[i][0] < t]

        logging.info("CLF none")

        return dirtyex

    # Define a data object to get the list of labels in the dataset
    data_object = Dataset(dataset_name)

    # preprocess data, especially featurization
    X_train_dirty, y_train_dirty, X_test_dirty, y_test_dirty = preprocess(dirty_df, [data_object.cfg.labels])
    X_train_gt, y_train_gt, X_test_gt, y_test_gt = preprocess(clean_df, [data_object.cfg.labels])

    # concatinate to get same format as activeclean
    X_dirty = vstack((X_train_dirty, X_test_dirty))
    #X_dirty = np.concatenate((X_train_dirty, X_test_dirty), axis=0)
    y_dirty = np.concatenate((y_train_dirty, y_test_dirty), axis=0)

    X_gt = vstack((X_train_gt, X_test_gt))
    #X_gt = np.concatenate((X_train_gt, X_test_gt), axis=0)
    y_gt = np.concatenate((y_train_gt, y_test_gt), axis=0)

    # get indices of dirty records/rows of training data
    indices_dirty = np.unique([row_i for (row_i, col_i), dummy in detections.items()])
    indices_clean = [i for i in range(0, X_train_dirty.shape[0]) if i not in indices_dirty]
    # indices_clean = [i for i in range(0, X_dirty.shape[0]) if i not in indices_dirty]

    # get indicies of splitted data
    N = X_dirty.shape[0]
    np.random.seed(1)
    idx = np.random.permutation(N)

    test_size = int(0.2 * N)
    train_size = N - test_size

    # get indices of train, test, val
    test_indices = idx[:test_size]
    train_indices = idx[test_size: N]
    clean_test_indices = translate_indices(test_indices, indices_clean)

    # initialize results dict
    results = {
        "model": 'ActiveClean',
        "S1": [],
        "S2": [0.0, 0.0, 0.0],
        "S3": [0.0, 0.0, 0.0],
        "S4": [],
        "S5": [0.0, 0.0, 0.0],
    }

    # S4: training: repaired/activeclean, test: gt
    p, r, f1 = activeclean_orig((X_dirty, y_dirty),
                                (X_gt, y_gt),
                                (X_gt[test_indices, :], y_gt[test_indices]),  # test on gt
                                # (X_gt[clean_test_indices,:], y_gt[clean_test_indices]),
                                X_dirty,  # X_full
                                (train_indices, indices_dirty, indices_clean),
                                total=int(sampling_budget * dirty_df.shape[0]))
    results["S4"] = [p, r, f1]

    # S1: training: dirty, test: dirty
    clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_dirty[train_indices, :], y_dirty[train_indices])

    average = "micro" if len(np.unique(y_dirty[test_indices])) > 2 else "binary"
    y_pred = clf.predict(X_dirty[test_indices, :])
    p, r, f1, _ = precision_recall_fscore_support(y_gt[test_indices], y_pred, average=average)
    results["S1"] = [p, r, f1]

    return results
