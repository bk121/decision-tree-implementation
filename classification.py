#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed.
##############################################################################

import numpy as np
from read_data import read_data
from node import Node
from evaluation_metrics import confusion_matrix, accuracy

files = ["train_full.txt", "train_sub.txt", "train_noisy.txt"]


class DecisionTreeClassifier(object):
    """Basic decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained

    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.root_node = None

    def fit(self, x, y):
        """Constructs a decision tree classifier from data

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(
            y
        ), "Training failed. x and y must have the same number of instances."
        self.root_node = Node(x, y)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def predict(self, x):
        """Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of test instances
                           K is the number of attributes

        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        for i, row in enumerate(x):
            predictions[i] = self.root_node.predict(row)

        # remember to change this if you rename the variable
        return predictions


x_sub, y_sub = read_data("data/train_sub.txt")
x_val, y_val = read_data("data/validation.txt")
classifier = DecisionTreeClassifier()
classifier.fit(x_sub, y_sub)

predictions = classifier.predict(x_val)

confusion = confusion_matrix(y_val, predictions)
print(accuracy(y_val, predictions))

#print(np.count_nonzero((y_val == predictions)) / y_val.size)
