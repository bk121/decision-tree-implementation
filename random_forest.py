from multiway_classification import MultiwayDecisionTreeClassifier
from binary_classification import BinaryDecisionTreeClassifier
from evaluation_metrics import accuracy, confusion_matrix, precision, recall, f1_score
from read_data import read_data
from random import sample, seed, choices
from statistics import mode
import numpy as np
import math


class RandomForest(object):
    def __init__(
        self, number_of_trees, decision_tree, div_param=8, max_branches=None
    ) -> None:
        self.number_of_trees = number_of_trees
        self.decision_tree = decision_tree
        self.div_param = div_param
        self.max_branches = max_branches

    def fit(self, x, y, x_val, y_val):
        num_of_attributes = np.shape(x)[1]
        # number of attributes is 10% of total possible
        num_of_attributes_in_each_tree = math.ceil(
            num_of_attributes / self.div_param)
        # create list of trees (forest)
        self.list_of_trees = []
        for i in range(self.number_of_trees):
            self.list_of_trees.append(self.decision_tree(self.max_branches))
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
            self.list_of_trees[i].fit(
                masked[sample_indices], y[sample_indices])
            # self.list_of_trees[i].prune(x_val, y_val)

    def predict(self, x):
        list_of_trees_predictions = np.empty(
            [self.number_of_trees, len(x)], dtype=str)

        for i in range(self.number_of_trees):
            preds = self.list_of_trees[i].predict(x)
            for j, pred in enumerate(preds):
                list_of_trees_predictions[i][j] = pred

        output = np.empty((len(list_of_trees_predictions[0, :])), dtype=str)

        for i in range(len(x)):
            output[i] = mode(list_of_trees_predictions[:, i])

        return output


x_full, y_full = read_data("data/train_full.txt")
x_test, y_test = read_data("data/test.txt")
x_val, y_val = read_data("data/validation.txt")


forest = RandomForest(
    50, MultiwayDecisionTreeClassifier, div_param=2, max_branches=3
)
forest.fit(x_full, y_full, x_val, y_val)
forest_predictions = forest.predict(x_test)
print(
    "\nAccuracy:\n",
    accuracy(y_test, forest_predictions)
)

forest = RandomForest(
    50, MultiwayDecisionTreeClassifier, div_param=3, max_branches=3
)
forest.fit(x_full, y_full, x_val, y_val)
forest_predictions = forest.predict(x_test)
print(
    "\nAccuracy:\n",
    accuracy(y_test, forest_predictions)
)

forest = RandomForest(
    50, MultiwayDecisionTreeClassifier, div_param=4, max_branches=3
)
forest.fit(x_full, y_full, x_val, y_val)
forest_predictions = forest.predict(x_test)
print(
    "\nAccuracy:\n",
    accuracy(y_test, forest_predictions)
)


# print("\n ----- RANDOM FOREST MB=2 ----- \n")
# params = np.linspace(1, 10, 10)
# mbs = np.linspace(2, 6, 5)
# print(params)
# print(mbs)
# for mb in mbs:
#     for param in params:
#         forest = RandomForest(
#             80, MultiwayDecisionTreeClassifier, div_param=param, max_branches=mb
#         )
#         forest.fit(x_full, y_full, x_val, y_val)
#         forest_predictions = forest.predict(x_test)
#         print(
#             "\nAccuracy:\n",
#             accuracy(y_test, forest_predictions),
#             "---------------",
#             param,
#             "---------------",
#             mb,
#         )


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
