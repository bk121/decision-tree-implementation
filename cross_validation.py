"""
File:           cross_validation.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Public Functions:   k_fold_split(n_splits, n_instances), 
                    train_test_k_fold(decision_tree_classifier, attributes, labels, n_splits)

Summary of File:

        Contains functions for performing cross-validation on classifier
"""
import numpy as np


def k_fold_split(n_splits, n_instances):
    # generate random permutation of [0...n_instances] (where n_instances == the total number
    # of data points in your data)
    shuffled_indices = np.random.default_rng().permutation(n_instances)

    # now I have one permutation of n_instances length which I need to partition
    # into n_splits. Note: We use array_split instead of .split() because we may need
    # our permutation into partitions of unequal length (not supported by split())
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(decision_tree_classifier, attributes, labels, n_splits):
    # Number of total instances to split
    n_instances = np.shape(attributes)[0]

    split_indices = k_fold_split(n_splits, n_instances)
    split_indices = np.array(split_indices)

    folds = []
    for i in range(n_splits):
        test_indices = split_indices[i]
        train_indices = np.delete(split_indices, i, 0).flatten()

        index_set = [train_indices, test_indices]

        folds.append(index_set)

    # Currently, folds[] has 10 rows (as per n_splits argument) and 2 columns;
    # first column is train_indices and 2nd is test_indices
    accuracies = []
    fitted_classifiers = []

    for i in range(n_splits):
        x_train = attributes[folds[i][0]]
        y_train = labels[folds[i][0]]

        decision_tree_classifier.fit(x_train, y_train)
        fitted_classifiers.append(decision_tree_classifier)

        x_test = attributes[folds[i][1]]
        y_test = labels[folds[i][1]]

        predictions = decision_tree_classifier.predict(x_test)

        accuracy = (np.count_nonzero(predictions == y_test)) / y_test.size
        accuracies.append(accuracy)

    return accuracies, fitted_classifiers