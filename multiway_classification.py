"""
File:           multiway_classification.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Classes:            MultiwayDecisionTreeClassifier
Public Functions:   fit(x,y), predict(x), prune(x_val,y_val)

Summary of File:

        Contains multiway decision tree classifier.
"""

import numpy as np
from itertools import combinations


class MultiwayDecisionTreeClassifier(object):
    """Multiway decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    max_branches (int): The maximum number of branches that can come from a node
    max_depth (int): Maximum number of layers to use in the tree
    root (Node): Root node of the decision tree
    node_count (int): How many nodes are in tree (including leaves)

    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self, max_depth=np.Inf, max_branches=None):
        """Constructor of decision tree class

        Args:
            max_depth (int, optional): Maximum depth the tree should reach
                                        Defaults to np.Inf.
            max_branches (int, optional): Maximum number of branches that can
                                        come from a node
                                        Defaults to None.
        """
        self.is_trained = False
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.root = None
        self.node_count = 0

    class Node(object):
        """Node of decision tree

        Attributes:
        leaf (bool): Indicates if a node is a leaf
        entropy (bool): Entropy of training data at node
        class_distribution (np.ndarray): Numpy array of shape (C,2)
                                         C is the number of unique classes
                                         [0] is clss labels
                                         [1] is frequency
        predicted class (char): Most common label in training data at node
        split_attribute (int): Attribute the node splits by
        split_values (list<int>): List of values to split at
        nodes (lits<Node>): list of Nodes coming from parent node
        """

        def __init__(self, entropy, class_distribution, predicted_class):
            """Constructor for a node

            Args:
                entropy (int): The entropy at the node
                class_distribution (np.ndarray): Numpy array of shape (C,2)
                                            C is the number of unique classes
                                            [0] is clss labels
                                            [1] is frequency
                predicted_class (char): Most common label in training data
            """
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
        # Build the decision tree
        self.node_count = 0
        self.root = self._build_tree(x, y)
        # Set a flag so that we know that the classifier has been trained
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
        # Check that the tree has been trained
        if not self.is_trained:
            print("Decision tree has not been trained")
            return
        # Call method to make single prediction for each item in array
        return np.array([self._predict(row) for row in x])

    def _predict(self, row):
        """Subfunction to make one prediction at a time

        Args:
            row (list<int>): Row of attribute data

        Returns:
            char: predicted classification
        """
        # Start at tree root
        node = self.root
        # Iterate while not at a leaf
        while not node.leaf:
            # Read split values
            split_vals = node.split_values
            node_number = 0  # Declare variable to find node
            # Iterate through split values
            for split_val in split_vals:
                # If the attribute is higher than the value
                if row[node.split_attribute] >= split_val:
                    # Must be in the next gap or higher
                    node_number += 1
            # Get correct node
            node = node.nodes[node_number]
        return node.predicted_class

    def prune(self, x, y):
        """Prune the tree with a validation data set

        Args:
            x (np.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of validation instances
                           K is the number of x
            y (np.ndarray): Class y, numpy array of shape (N, )
                           Each element in y is a str
        """
        # Start at the root of the tree
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
        # Check if all nodes downstream are leaves
        if all(n.leaf for n in node.nodes):
            # Get accuracy reading
            prior_acc = self._accuracy(x, y)
            # Remove the leaves
            nodes = node.nodes
            node.nodes = None
            node.leaf = True
            # Get another accuracy reading
            after_acc = self._accuracy(x, y)
            self.node_count -= len(nodes)
            # Compare accuracies
            if after_acc <= prior_acc:
                # Undo if bad
                node.nodes = nodes
                self.node_count += len(nodes)
                node.leaf = False
            return
        # Iterate through all nodes in tree
        for n in node.nodes:
            if not n.leaf:
                self._reccursively_prune(x, y, n)

    def _accuracy(self, x, y):
        """Get the accuracy of the tree

        Args:
            x (np.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of validation instances
                           K is the number of x
            y (np.ndarray): Class y, numpy array of shape (N, )
                           Each element in y is a str

        Returns:
            int: accuracy of the tree
        """
        # Make predictions
        preds = self.predict(x)
        try:
            # Compare to gold standard
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
        # Get counts of labels
        unique_labels, counts = np.unique(y, return_counts=True)
        # Make each a proportion of total
        p_x = counts / y.size
        # Apply entropy formula
        entropy = -np.sum(p_x * np.log2(p_x))
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
            # Get entropies of groups
            entropies.append(self._evaluate_entropy(y_sub))
            # Get sizes of groups
            sizes.append(y_sub.size)
        # Get total size
        n_total = np.sum(sizes)
        # Apply information gain formula
        ig = current_entropy
        for n, ent in zip(sizes, entropies):
            ig -= (n / n_total) * ent
        return ig

    def _split_data(self, split_attr, split_vals, x, y):
        """Splits the data for a given attribute and value

        Args:
            split_attr (int): Attribute to split data by
            split_values (list<int>): List of values to split at
            x (numpy.ndarray): Instances, numpy array of shape (N, K)
                               N is the number of instances
                               K is the number of x
            y (numpy.ndarray): Class y, numpy array of shape (N, )
                               Each element in y is a str

        Returns:
            tuple: Tuple of np.ndarrays representing the divided datasets where
                    first is < firest value etc.
                    (output_x, output_y)
        """
        # Set up output arrays
        output_x = []
        output_y = []
        for split_val in split_vals:
            # Get indices in bracket
            indices = x[:, split_attr] < split_val
            output_x.append(x[indices])
            # Remove data from x
            x = x[~indices]
            output_y.append(y[indices])
            y = y[~indices]
        # Add what's left in last position
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
            tuple: tuple of integers and a list representing optimal split
                    attribute and optimal split values respectively
        """
        # Check there is some data
        if x.size <= 1:
            return None, None
        # Get the current entropy
        current_entropy = self._evaluate_entropy(y)
        max_IG_attribute = None
        max_IG_split_vals = None
        max_IG = -1

        # Test all attributes
        for split_attr in range(np.shape(x)[1]):
            # Get possible split values
            possible_split_vals = np.unique(sorted(x[:, split_attr]))
            # Use max split value if set
            max_branches = (
                self.max_branches if self.max_branches else len(possible_split_vals)
            )
            # Try all number of branches
            for n in range(max_branches - 1):
                # Test all the combinations of split values
                for split_vals in combinations(possible_split_vals, n + 1):
                    # Make split
                    split_x, split_y = self._split_data(split_attr, split_vals, x, y)
                    # Get the gain of the split
                    information_gain = self._evaluate_information_gain(
                        current_entropy, split_y
                    )
                    # If it has better performance, update
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
        # Get the labels and counts in data
        classes, counts = np.unique(y, return_counts=True)
        # Find the mode for predicition
        predicted_class = classes[np.argmax(counts)]
        # Get the classes at node
        class_dist = np.asarray((classes, counts)).T
        # Set up a new node
        node = self.Node(
            entropy=self._evaluate_entropy(y),
            class_distribution=class_dist,
            predicted_class=predicted_class,
        )
        self.node_count += 1
        # Check ending conditions
        if depth < self.max_depth and len(np.unique(y)) > 1 and not (x == x[0]).all():
            # Get the best split
            split_attr, split_vals = self._find_best_split(x, y)
            if split_attr != None:
                # Get the split up data
                split_x, split_y = self._split_data(split_attr, split_vals, x, y)
                # Check none of the subsets are 0 length
                for check_x in split_x:
                    if len(check_x) == 0:
                        node.leaf = True
                        return node
                # Put correvct data in leaves
                node.split_values = split_vals
                node.split_attribute = split_attr
                node.nodes = []
                for x, y in zip(split_x, split_y):
                    # Reccursive call
                    node.nodes.append(self._build_tree(x, y, depth + 1))
        else:
            # If end condition met set leaf to true
            node.leaf = True
        return node
