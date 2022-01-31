import numpy as np
from read_data import read_data
from leaf import Leaf


class Node:
    def __init__(self, attributes, labels):
        self.split_attr, self.split_val = self.__find_best_split(attributes, labels)

        left_attr, right_attr, left_labels, right_labels = self.__get_split(
            attributes, labels, self.split_attr, self.split_val
        )

        self.left_branch = (
            Node(left_attr, left_labels)
            if left_labels.size > 400
            else Leaf(left_labels)
        )
        self.right_branch = (
            Node(right_attr, right_labels)
            if right_labels.size > 400
            else Leaf(right_labels)
        )

    def predict(self, data):
        if data[self.split_attr] < self.split_val:
            return self.left_branch.predict(data)
        else:
            return self.right_branch.predict(data)

    def __evaluate_entropy(self, labels):
        total = 0
        unique_labels, counts = np.unique(labels, return_counts=True)
        p_x = counts / labels.size
        for p in p_x:
            total += p * np.log2(p)
        return -total

    def __evaluate_information_gain(self, attributes, labels, split_attr, split_val):
        left_attr, right_attr, left_labels, right_labels = self.__get_split(
            attributes, labels, split_attr, split_val
        )

        full_data_entropy = self.__evaluate_entropy(labels)
        left_data_entropy = self.__evaluate_entropy(left_labels)
        right_data_entropy = self.__evaluate_entropy(right_labels)

        ig = (
            full_data_entropy
            - left_labels.size / labels.size * left_data_entropy
            - right_labels.size / labels.size * right_data_entropy
        )

        return ig

    def __find_best_split(self, attributes, labels):
        max_IG_entropy = 0
        max_IG_attribute = 0
        max_IG_split_val = 0

        for split_attr in range(np.shape(attributes)[1]):
            for split_val in range(
                min(attributes[:, split_attr]), max(attributes[:, split_attr]) + 1
            ):
                information_gain = self.__evaluate_information_gain(
                    attributes, labels, split_attr, split_val
                )
                if information_gain > max_IG_entropy:
                    max_IG_entropy = information_gain
                    max_IG_attribute = split_attr
                    max_IG_split_val = split_val

        return (max_IG_attribute, max_IG_split_val)

    def __get_split(self, attributes, labels, split_attr, split_val):
        left_indices = attributes[:, split_attr] < split_val
        right_indices = attributes[:, split_attr] >= split_val
        left_attr = attributes[left_indices]
        right_attr = attributes[right_indices]
        left_labels = labels[left_indices]
        right_labels = labels[right_indices]

        return (left_attr, right_attr, left_labels, right_labels)
