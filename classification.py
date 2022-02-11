"""
File:           binary_classification.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Classes:            MultiwayDecisionTreeClassifier
Public Functions:   fit(x,y), predict(x), prun(x_val,y_val)

Summary of File:

        Contains node class for binary decision tree classifier.
"""

import numpy as np


class DecisionTreeClassifier(object):
    """Basic decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    max_depth (int): Maximum number of layers to use in the tree
    root (Node): Root node of the decision tree

    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self, max_depth=np.Inf):
        """Constructor of decision tree class

        Args:
            max_depth (int, optional): maximum depth the tree should reach.
                                        Defaults to np.Inf.
        """
        self.is_trained = False
        self.max_depth = max_depth
        self.root = None
        self.node_count = 0

    class Node(object):
        """Node of decision tree

        Attributes:
        entropy (bool): Entropy of training data at node
        class_distribution (np.ndarray): Numpy array of shape (C,2)
                                         C is the number of unique classes
                                         [0] is clss labels
                                         [1] is frequency
        predicted class (int): Most common label in training data at node
        split_attribute (int): Attribute the node splits by
        split _value (int): Value the node splits at
        left_branch (Node): Left node (less than value)
        right_branch(Node): Right node (more than/equal to value)
        """

        def __init__(self, entropy, class_distribution, predicted_class):
            self.leaf = False
            self.entropy = entropy
            self.class_distribution = class_distribution
            self.predicted_class = predicted_class
            self.split_attribute = 0
            self.split_value = 0
            self.left_branch = None
            self.right_branch = None

    def fit(self, x, y):
        """Constructs a decision tree classifier from data

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
                           N is the number of instances
                           K is the number of x
        y (numpy.ndarray): Class y, numpy array of shape (N, )
                           Each element in y is a str
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(
            y
        ), "Training failed. x and y must have the same number of instances."
        self.root = self._build_tree(x, y)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def predict(self, x):
        """Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of test instances
                           K is the number of x

        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        if not self.is_trained:
            print("Decision tree has not been trained")
            return

        def _predict(row):
            node = self.root
            while node.left_branch:
                node = (
                    node.left_branch
                    if row[node.split_attribute] < node.split_value
                    else node.right_branch
                )
            return node.predicted_class

        return np.array([_predict(row) for row in x])

    def prune(self, x, y):
        """Prune the tree with a validation data set

        Args:
            x (np.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of validation instances
                           K is the number of x
            y (np.ndarray): Class y, numpy array of shape (N, )
                           Each element in y is a str
        """
        node = self.root
        self._reccursively_prune(x, y, node)

    def _reccursively_prune(self, x, y, node):
        """Recurrsive function to prune the tree

        Args:
            x (np.ndarray): Instances, numpy array of shape (M, K)
                        M is the number of validation instances
                        K is the number of x
            y (np.ndarray): Class y, numpy array of shape (N, )
                        Each element in y is a str
            node (Node): Node to move from in pruning
        """
        if node.left_branch.leaf and node.right_branch.leaf:
            prior_acc = self._accuracy(x, y)
            left_branch = node.left_branch
            right_branch = node.right_branch
            node.left_branch = None
            node.right_branch = None
            node.leaf = True
            after_acc = self._accuracy(x, y)
            self.node_count -= 2
            if after_acc <= prior_acc:
                node.left_branch = left_branch
                node.right_branch = right_branch
                self.node_count += 2
            return
        if not node.left_branch.leaf:
            self._reccursively_prune(x, y, node.left_branch)
        if not node.right_branch.leaf:
            self._reccursively_prune(x, y, node.right_branch)

    def _accuracy(self, x, y):
        preds = self.predict(x)
        try:
            return np.sum(y == preds) / len(y)
        except ZeroDivisionError:
            return 0

    def _evaluate_entropy(self, y):
        """Evaluates the entropy of a dataset

        Args:
            y (numpy.ndarray): Class y, numpy array of shape (N, )
                               Each element in y is a str

        Returns:
            int: entropy of the data
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        p_x = counts / y.size
        entropy = -np.sum(p_x * np.log2(p_x))
        return entropy

    def _evaluate_information_gain(self, current_entropy, y_left, y_right):
        """Evaluates the information gain of a specified split

        Args:
            current_entropy (int): Current entropy of the data
            y_left (numpy.ndarray): Subset of label data
            y_right (numpy.ndarray): Subset of label data

        Returns:
            int: Information gain of the division
        """
        left_data_entropy = self._evaluate_entropy(y_left)
        right_data_entropy = self._evaluate_entropy(y_right)
        n_left = y_left.size
        n_right = y_right.size
        n_total = n_left + n_right
        ig = (
            current_entropy
            - (n_left / n_total) * left_data_entropy
            - (n_right / n_total) * right_data_entropy
        )
        return ig

    def _split_data(self, split_attr, split_val, x, y):
        """Splits the data for a given attribute and value

        Args:
            split_attr (int): Attribute to split data by
            split_val (int): Value to split data at
            x (numpy.ndarray): Instances, numpy array of shape (N, K)
                               N is the number of instances
                               K is the number of x
            y (numpy.ndarray): Class y, numpy array of shape (N, )
                               Each element in y is a str

        Returns:
            tuple: Tuple of np.ndarrays representing the divided left and right
                    datasets where left is < value and right is >= value.
                    (x_left, x_right, y_left, y_right)
        """
        left_indices = x[:, split_attr] < split_val
        x_left = x[left_indices]
        x_right = x[~left_indices]
        y_left = y[left_indices]
        y_right = y[~left_indices]
        return x_left, x_right, y_left, y_right

    def _find_best_split(self, x, y):
        """Function to find the best data split by information gain

        Args:
            x (numpy.ndarray): Instances, numpy array of shape (N, K)
                               N is the number of instances
                               K is the number of x
            y (numpy.ndarray): Class y, numpy array of shape (N, )
                               Each element in y is a str

        Returns:
            tuple: tuple of integers representing optimal split attribute and
                   optimal split value respectively
        """
        if x.size <= 1:
            return None, None
        current_entropy = self._evaluate_entropy(y)
        max_IG_attribute = None
        max_IG_split_val = None
        max_IG = -1
        for split_attr in range(np.shape(x)[1]):
            for split_val in np.unique(sorted(x[:, split_attr])):
                x_left, x_right, y_left, y_right = self._split_data(
                    split_attr, split_val, x, y
                )
                information_gain = self._evaluate_information_gain(
                    current_entropy, y_left, y_right
                )
                if information_gain > max_IG:
                    max_IG = information_gain
                    max_IG_attribute = split_attr
                    max_IG_split_val = split_val
        return (max_IG_attribute, max_IG_split_val)

    def _build_tree(self, x, y, depth=0):
        """Build the decision tree reccursively

        Args:
            x (numpy.ndarray): Instances, numpy array of shape (N, K)
                               N is the number of instances
                               K is the number of x
            y (numpy.ndarray): Class y, numpy array of shape (N, )
                               Each element in y is a str
            depth (int, optional): Current depth of the tree. Defaults to 0.

        Returns:
            Node: Root node of the tree
        """
        classes, counts = np.unique(y, return_counts=True)
        predicted_class = y[np.argmax(counts)]
        class_dist = np.asarray((classes, counts)).T
        node = self.Node(
            entropy=self._evaluate_entropy(y),
            class_distribution=class_dist,
            predicted_class=predicted_class,
        )
        self.node_count += 1
        if depth < self.max_depth and len(np.unique(y)) > 1:
            split_attr, split_val = self._find_best_split(x, y)
            if split_attr != None:
                x_left, x_right, y_left, y_right = self._split_data(
                    split_attr, split_val, x, y
                )
                if len(x_left) == 0 or len(x_right) == 0:
                    node.leaf = True
                    return node
                node.split_value = split_val
                node.split_attribute = split_attr
                node.left_branch = self._build_tree(x_left, y_left, depth + 1)
                node.right_branch = self._build_tree(x_right, y_right, depth + 1)
        else:
            node.leaf = True
        return node
