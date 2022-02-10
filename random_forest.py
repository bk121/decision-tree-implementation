"""
File:           random_forest.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Classes:            RandomForest(number_of_trees, decision_tree,  //
                                    div_param=8, max_branches=None)
Public Functions:   fit(x, y, x_val, y_val), predict(x)

Summary of File:

        Contains random decision forest class.
"""

from random import sample, seed, choices
from statistics import mode
import numpy as np
import math


class RandomForest(object):
    """
    Class for building and running random forests

    Attributes:
    number_of_trees (int): Number of trees in the forest
    decision_tress (Tree): Type of tree to use
    div_param (int): Divider of attrubutes
    max_branches (int): (optional) max branches per node
    """

    def __init__(
        self, number_of_trees, decision_tree, div_param=4, max_branches=None
    ) -> None:
        """
        Constructor for RDF

        Args:
            number_of_trees (int): Number of trees in the forest
            decision_tress (Tree): Type of tree to use
            div_param (int): Divider of attrubutes
            max_branches (int): (optional) max branches per node
        """
        self.number_of_trees = number_of_trees
        self.decision_tree = decision_tree
        self.div_param = div_param
        self.max_branches = max_branches

    def fit(self, x, y, x_val, y_val):
        """
        Trains the RDF

        Args:
            x (numpy.ndarray): Instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of x
            y (numpy.ndarray): Class y, numpy array of shape (N, )
                            Each element in y is a str
            x_val (np.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of validation instances
                           K is the number of x
            y_val (np.ndarray): Class y, numpy array of shape (N, )
                           Each element in
        """
        num_of_attributes = np.shape(x)[1]
        # number of attributes is 10% of total possible
        num_of_attributes_in_each_tree = math.ceil(num_of_attributes / self.div_param)
        # create list of trees (forest)
        self.list_of_trees = []
        for i in range(self.number_of_trees):
            if self.max_branches:
                self.list_of_trees.append(self.decision_tree(self.max_branches))
            else:
                self.list_of_trees.append(self.decision_tree())
            seed()
            attributes_subset = sample(
                (range(num_of_attributes)),
                num_of_attributes - num_of_attributes_in_each_tree,
            )
            sample_indices = choices(range(len(x)), k=len(x))
            temp = np.zeros_like(x)
            temp[:, attributes_subset] = 1
            masked = np.ma.masked_array(x, temp)
            masked = np.ma.filled(masked, 100)
            self.list_of_trees[i].fit(masked[sample_indices], y[sample_indices])
            self.list_of_trees[i].prune(x_val, y_val)

    def predict(self, x):
        """
        Makes predictions with the forest

        Args:
            x (np.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of validation instances
                           K is the number of x

        Returns:
            np.ndarray: Predictions
        """
        list_of_trees_predictions = np.empty([self.number_of_trees, len(x)], dtype=str)

        for i in range(self.number_of_trees):
            preds = self.list_of_trees[i].predict(x)
            for j, pred in enumerate(preds):
                list_of_trees_predictions[i][j] = pred

        output = np.empty((len(list_of_trees_predictions[0, :])), dtype=str)

        for i in range(len(x)):
            output[i] = mode(list_of_trees_predictions[:, i])

        return output
