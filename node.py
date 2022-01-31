import numpy as np
from read_data import read_data
from leaf import Leaf


class Node:
    def __init__(self, attributes, labels):
        self.split_attr, self.split_val = self.__find_best_split(attributes, labels)

        left_data, right_data, left_labels, right_labels = self.__get_split(
            attributes, labels, self.split_attr, self.split_val
        )
        print(np.shape(left_data))
        print(np.shape(right_data))

        self.left_branch = (
            Node(left_data, left_labels)
            if left_labels.size > 400
            else Leaf(left_labels)
        )
        self.right_branch = (
            Node(right_data, right_labels)
            if right_labels.size > 400
            else Leaf(right_labels)
        )

    def predict(self, data):
        if data[self.split_attr] < self.split_val:
            return self.left_branch.predict(data)
        else:
            return self.right_branch.predict(data)

    def __evaluate_entropy(self, labels):
        labels = np.unique(labels)
        total = 0
        labels, counts = np.unique(labels, return_counts=True)
        p_x = counts / labels.size
        print(p_x)
        for p in p_x:
            total += p * np.log2(p)
        print(total)
        return -total

    def __evaluate_information_gain(self, attributes, labels, split_val, split_attr):
        left_data, right_data, left_labels, right_labels = self.__get_split(
            attributes, labels, split_val, split_attr
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

        for attribute in range(np.shape(attributes)[1]):
            for split_val in range(
                min(attributes[:, attribute]), max(attributes[:, attribute]) + 1
            ):
                information_gain = self.__evaluate_information_gain(
                    attributes, labels, split_val, attribute
                )
                if information_gain > max_IG_entropy:
                    max_IG_entropy = information_gain
                    max_IG_attribute = attribute
                    max_IG_split_val = split_val

        return (max_IG_attribute, max_IG_split_val)

    def __get_split(self, attributes, labels, split_val, split_attr):
        left_indices = attributes[:, split_attr] < split_val
        right_indices = attributes[:, split_attr] >= split_val
        left_data = attributes[left_indices]
        right_data = attributes[right_indices]
        # (i for i in range(labels.size) if i not in left_indices)
        left_labels = labels[left_indices]
        right_labels = labels[right_indices]

        return (left_data, right_data, left_labels, right_labels)
