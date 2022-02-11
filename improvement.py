"""
File:           improvement.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Public Functions:   train_and_predict(x_train, y_train, x_test, x_val, y_val)

Summary of File:

        Contains the improvements made to the decision tree classifier.
"""

import numpy as np
from multiway_classification import MultiwayDecisionTreeClassifier
from classification import DecisionTreeClassifier
from random_forest import RandomForest


def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """Interface to train and test the new/improved decision tree.

    This function is an interface for training and testing the new/improved
    decision tree classifier.

    x_train and y_train should be used to train your classifier, while
    x_test should be used to test your classifier.
    x_val and y_val may optionally be used as the validation dataset.
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K)
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K)
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K)
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    """

    classifier = RandomForest(150, DecisionTreeClassifier, 2)
    classifier.fit(x_train, y_train, x_val, y_val)
    predictions = classifier.predict(x_test)
    return predictions
