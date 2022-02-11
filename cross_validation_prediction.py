"""
File:           cross_validation_prediction.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Public Functions:  

Summary of File: cross_validation_prediction(decision_tree_classifier, train_attr, train_labels, test_attr, n_splits)

        Contains function for making predictions by combining the predictions of the decision trees produces in 
        the cross-validation process
"""

import numpy as np
from cross_validation import train_test_k_fold
from evaluation_metrics import accuracy
from classification import DecisionTreeClassifier
from read_data import read_data


def cross_validation_prediction(
    decision_tree_classifier, train_attr, train_labels, test_attr, n_splits
):

    # Add cross_validation function call here
    accuracies, fitted_classifiers = train_test_k_fold(
        decision_tree_classifier, train_attr, train_labels, n_splits
    )

    # Create list of lists with predictions from each of the n classifiers
    predictions_array = []
    for i in range(n_splits):
        predictions_array.append(fitted_classifiers[i].predict(test_attr))

    # For each row, add to final_predictions the mode of the n classifiers' predictions for that given row
    final_predictions = []
    for i in range(len(test_attr)):
        row_predictions = [prediction[i] for prediction in predictions_array]
        row_predictions_mode = max(set(row_predictions), key=row_predictions.count)
        final_predictions.append(row_predictions_mode)

    return final_predictions