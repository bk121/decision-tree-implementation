"""
File:           classification.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 02/02/2022
Last Edit By:   Ted Jenks

Classes:            DecisionTreeClassifier
Public Functions:   fit(x,y), predict(x)

Summary of File:

        Contains node class for binary decision tree classifier.
"""

from pyexpat.errors import XML_ERROR_INVALID_TOKEN
import numpy as np
from itertools import combinations
from read_data import read_data
from random import sample, randint, seed, choices
import math
from statistics import mode
from evaluation_metrics import accuracy, confusion_matrix, precision, recall, f1_score
import pickle


class DecisionTreeClassifier(object):
    """Basic decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    max_depth (int): Maximum number of layers to use in the tree
    root (Node): Root node of the decision tree

    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    # prune(x_val, y_val): Post-prunes the decision tree < -- NOT YET
    """

    def __init__(self, max_depth=np.Inf, max_branches=None):
        """Constructor of decision tree class

        Args:
            max_depth (int, optional): maximum depth the tree should reach.
                                        Defaults to np.Inf.
        """
        self.is_trained = False
        self.max_branches = max_branches
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
            self.split_values = []
            self.nodes = None

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
            while not node.leaf:
                split_vals = node.split_values
                outer = 0
                for split_val in split_vals:
                    if row[node.split_attribute] >= split_val:
                        outer += 1
                node = node.nodes[outer]
            return node.predicted_class

        return np.array([_predict(row) for row in x])

    def prune(self, x, y):
        node = self.root
        self._reccursively_prune(x, y, node)

    def _reccursively_prune(self, x, y, node):
        if all(n.leaf for n in node.nodes):
            prior_acc = self._accuracy(x, y)
            nodes = node.nodes
            node.nodes = None
            node.leaf = True
            after_acc = self._accuracy(x, y)
            self.node_count -= len(nodes)
            if after_acc <= prior_acc:
                node.nodes = nodes
                self.node_count += len(nodes)
                node.leaf = False
            return
        for n in node.nodes:
            if not n.leaf:
                self._reccursively_prune(x, y, n)

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
        entropy = -np.sum(p_x * np.log2(p_x + 1e-20))
        return entropy

    def _evaluate_information_gain(self, current_entropy, split_y):
        """Evaluates the information gain of a specified split

        Args:
            current_entropy (int): Current entropy of the data
            y_left (numpy.ndarray): Subset of label data
            y_right (numpy.ndarray): Subset of label data

        Returns:
            int: Information gain of the division
        """
        entropies = []
        sizes = []
        for y_sub in split_y:
            entropies.append(self._evaluate_entropy(y_sub))
            sizes.append(y_sub.size)
        n_total = np.sum(sizes)
        ig = current_entropy
        for n, ent in zip(sizes, entropies):
            ig -= (n / n_total) * ent
        return ig

    def _split_data(self, split_attr, split_vals, x, y):
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
        output_x = []
        output_y = []
        for split_val in split_vals:
            indices = x[:, split_attr] < split_val
            output_x.append(x[indices])
            x = x[~indices]
            output_y.append(y[indices])
            y = y[~indices]
        output_x.append(x)
        output_y.append(y)
        return output_x, output_y

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
        max_IG_split_vals = None
        max_IG = -1

        for split_attr in range(np.shape(x)[1]):
            possible_split_vals = np.unique(sorted(x[:, split_attr]))
            max_branches = (
                self.max_branches if self.max_branches else len(possible_split_vals)
            )
            for n in range(max_branches):
                for split_vals in combinations(possible_split_vals, n + 1):
                    split_x, split_y = self._split_data(split_attr, split_vals, x, y)
                    information_gain = self._evaluate_information_gain(
                        current_entropy, split_y
                    )
                    if information_gain > max_IG:
                        max_IG = information_gain
                        max_IG_attribute = split_attr
                        max_IG_split_vals = split_vals
        return (max_IG_attribute, max_IG_split_vals)

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
        if depth < self.max_depth and len(np.unique(y)) > 1 and not (x == x[0]).all():
            split_attr, split_vals = self._find_best_split(x, y)
            if split_attr != None:
                split_x, split_y = self._split_data(split_attr, split_vals, x, y)
                for check_x in split_x:
                    if len(check_x) == 0:
                        node.leaf = True
                        return node
                node.split_values = split_vals
                node.split_attribute = split_attr
                node.nodes = []
                for x, y in zip(split_x, split_y):
                    node.nodes.append(self._build_tree(x, y, depth + 1))
        else:
            node.leaf = True
        return node


class RandomForest(object):
    def fit(self, x, y, x_val, y_val, number_of_trees, max_branches=None):
        self.x = x
        self.y = y
        self.number_of_trees = number_of_trees
        self.max_branches = max_branches
        num_of_attributes = np.shape(x)[1]
        # number of attributes is 10% of total possible
        num_of_attributes_in_each_tree = math.ceil(num_of_attributes / 10)
        # create list of trees (forest)
        self.list_of_trees = np.full(
            number_of_trees, DecisionTreeClassifier(max_branches=self.max_branches)
        )
        self.attributes = []
        for i in range(number_of_trees):
            # attributes_subset = sample(
            #     (range(num_of_attributes)),
            #     num_of_attributes - num_of_attributes_in_each_tree,
            # )
            # temp = np.zeros_like(x)
            # temp[:, attributes_subset] = 1
            # masked = np.ma.masked_array(x, temp)
            # masked = np.ma.filled(masked, 0)
            # self.list_of_trees[i].fit(masked, y)
            # self.list_of_trees[i].prune(x_val, y_val)

            seed()
            attributes_subset = sample(
                (range(np.shape(x)[1])), num_of_attributes_in_each_tree
            )
            attributes_subset = np.asarray(attributes_subset)
            self.attributes.append(attributes_subset)
            # cut down to just chosen columns
            x_sample = x[:, attributes_subset]

            self.list_of_trees[i].fit(x_sample, y)

    def predict(self, x):

        list_of_trees_predictions = np.empty([self.number_of_trees, len(x)], dtype=str)

        for i in range(self.number_of_trees):
            preds = self.list_of_trees[i].predict(x[:, self.attributes[i]])
            for j, pred in enumerate(preds):
                list_of_trees_predictions[i][j] = pred

        output = np.empty(len(x), dtype=str)

        for i in range(len(x)):
            output[i] = mode(list_of_trees_predictions[:, i])

        return output


x_full, y_full = read_data("data/train_full.txt")
x_test, y_test = read_data("data/test.txt")
x_val, y_val = read_data("data/validation.txt")

print("\n ----- RANDOM FOREST MB=2 ----- \n")
forest = RandomForest()
forest.fit(x_full, y_full, x_val, y_val, 128, max_branches=2)
forest_predictions = forest.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, forest_predictions))
print("\nAccuracy:\n", accuracy(y_test, forest_predictions))
# print("\nPrecision:\n", precision(y_test, forest_predictions))
# print("\nRecall:\n", recall(y_test, forest_predictions))
# print("\nF1_Score:\n", f1_score(y_test, forest_predictions))

# save_classifiers = open(
#     "trained_classifiers/rf_pruned_two_way.pickle",
#     "wb",
# )
# pickle.dump(forest, save_classifiers)
# save_classifiers.close()

# print("\n ----- RANDOM FOREST MB=3 ----- \n")
# forest = RandomForest()
# forest.fit(x_full, y_full, x_val, y_val, 128, max_branches=3)
# forest_predictions = forest.predict(x_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, forest_predictions))
# print("\nAccuracy:\n", accuracy(y_test, forest_predictions))

# save_classifiers = open(
#     "trained_classifiers/rf_pruned_three_way.pickle",
#     "wb",
# )
# pickle.dump(forest, save_classifiers)
# save_classifiers.close()

# print("\n ----- RANDOM FOREST MB=4 ----- \n")
# forest = RandomForest()
# forest.fit(x_full, y_full, x_val, y_val, 128, max_branches=4)
# forest_predictions = forest.predict(x_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, forest_predictions))
# print("\nAccuracy:\n", accuracy(y_test, forest_predictions))

# save_classifiers = open(
#     "trained_classifiers/rf_pruned_four_way.pickle",
#     "wb",
# )
# pickle.dump(forest, save_classifiers)
# save_classifiers.close()

# print("\n ----- RANDOM FOREST MB=5 ----- \n")
# forest = RandomForest()
# forest.fit(x_full, y_full, x_val, y_val, 128, max_branches=5)
# forest_predictions = forest.predict(x_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, forest_predictions))
# print("\nAccuracy:\n", accuracy(y_test, forest_predictions))

# save_classifiers = open(
#     "trained_classifiers/rf_pruned_five_way.pickle",
#     "wb",
# )
# pickle.dump(forest, save_classifiers)
# save_classifiers.close()

# print("\n ----- RANDOM FOREST MB=inf ----- \n")
# forest = RandomForest()
# forest.fit(x_full, y_full, x_val, y_val, 128)
# forest_predictions = forest.predict(x_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, forest_predictions))
# print("\nAccuracy:\n", accuracy(y_test, forest_predictions))

# save_classifiers = open(
#     "trained_classifiers/rf_pruned_multiway.pickle",
#     "wb",
# )
# pickle.dump(forest, save_classifiers)
# save_classifiers.close()

print("\n ------- SIMPLE BINARY TREE ------- \n")
classifier = DecisionTreeClassifier(max_branches=2)
classifier.fit(x_full, y_full)
predictions = classifier.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nAccuracy:\n", accuracy(y_test, predictions))
# print("\nPrecision:\n", precision(y_test, predictions))
# print("\nRecall:\n", recall(y_test, predictions))
# print("\nF1_Score:\n", f1_score(y_test, predictions))
# print("\nNode Count:\n", classifier.node_count)

print("\n ------- PRUNED BINARY TREE ------- \n")
classifier.prune(x_val, y_val)
predictions = classifier.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nAccuracy:\n", accuracy(y_test, predictions))
# print("\nPrecision:\n", precision(y_test, predictions))
# print("\nRecall:\n", recall(y_test, predictions))
# print("\nF1_Score:\n", f1_score(y_test, predictions))
# print("\nNode Count:\n", classifier.node_count)
