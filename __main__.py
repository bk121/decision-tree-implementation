"""
File:           __main__.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 11/02/2022
Last Edit By:   Ben Kirwan

Summary of File:

        Contains main file to run classifier(s).
"""
import numpy as np
from cross_validation_prediction import cross_validation_prediction
from classification import DecisionTreeClassifier
from random_forest import RandomForest
from multiway_classification import MultiwayDecisionTreeClassifier
from read_data import read_data
from evaluation_metrics import accuracy, confusion_matrix, precision, recall, f1_score
from cross_validation import train_test_k_fold

if __name__ == "__main__":

    print("\n\n-------------Examining the dataset--------------\n\n")

    files = ["train_full.txt", "train_sub.txt", "train_noisy.txt"]

    x_full, y_full = read_data("data/" + files[0])
    x_sub, y_sub = read_data("data/" + files[1])
    x_noisy, y_noisy = read_data("data/" + files[2])
    x_val, y_val = read_data("data/validation.txt")
    x_test, y_test = read_data("data/test.txt")

    x_array = [x_full, x_sub, x_noisy]
    y_array = [y_full, y_sub, y_noisy]

    print("\nDATA SHAPES")
    print("-----------\n")

    for file, x, y in zip(files, x_array, y_array):
        print(" ", file, ":")
        print("      Attribute array shape (instances, attributes) :", np.shape(x))
        print("      Class label array shape (instances) :", np.shape(y), "\n")

    print("\nDATA CLASSES")
    print("------------\n")

    classes = np.unique(y_full)
    proportions_plot = ["", "", ""]
    for j, (file, x, y) in enumerate(zip(files, x_array, y_array)):
        proportion = np.empty(np.size(classes))
        for i, label in enumerate(classes):
            proportion[i] = np.count_nonzero(y == label) / np.size(y)
        proportions_plot[j] = proportion
        class_proportion = np.vstack((classes, proportion)).T
        print("\n ", file, ":")
        print("\n Classes Proportions\n", class_proportion)

    print("\nDATA RANGES")
    print("------------\n")

    range_plot = ["", "", ""]
    for j, (file, x, y) in enumerate(zip(files, x_array, y_array)):
        print("\n ", file, ":")
        x_t = x.T
        ranges = np.empty(np.shape(x_full)[1])
        for i, column in enumerate(x_t):
            print(
                "      Range of values in attribute",
                i,
                ":",
                np.max(column) - np.min(column),
                "\t with max: ",
                np.max(column),
                " and min: ",
                np.min(column),
            )
            ranges[i] = np.max(column) - np.min(column)
        range_plot[j] = ranges

    print("\nNOISY/FULL COMPARISON")
    print("---------------------\n")

    print("\n  Proportion of shared classes in noisy/full :")

    crossover = 0
    for row, val in zip(x_full, y_full):
        ind = np.where((x_noisy == row).all(axis=1))
        if val == y_noisy[ind[0][0]]:
            crossover += 1

    proportion = crossover / np.size(y_array[0])

    print("     ", proportion)

    print("\n\n-------------Implementing A Decision Tree--------------\n\n")
    print("\nTraining binary classifiers...\n")
    bin_trees = []
    data_files = ["data/" + file for file in files]
    for data_file in data_files:
        x,y=read_data(data_file)
        bin_tree=DecisionTreeClassifier()
        bin_tree.fit(x,y)
        bin_trees.append(bin_tree)
    print("\nBinary classifiers trained!\n")

    print("\n\n-------------Evaluation Metrics--------------\n\n")

    print("\nConfusion Matrices:\n")
    for data_file, tree in zip(data_files, bin_trees):
        print(
            "\nConfusion matrix for Binary decision tree (Trained on "
            + data_file
            + "):"
        )
        print(confusion_matrix(y_test, tree.predict(x_test)))

    print("\n\n\nMetrics per label for each decision tree:")
    macros = []
    for tree, file in zip(bin_trees, files):
        tree_prediction = tree.predict(x_test)
        letters = np.array(classes)
        tree_precision = np.around(precision(y_test, tree_prediction)[0], 3)
        macro_precision = np.around(precision(y_test, tree_prediction)[1], 3)
        tree_recall = np.around(recall(y_test, tree_prediction)[0], 3)
        macro_recall = np.around(recall(y_test, tree_prediction)[1], 3)
        tree_f1_score = np.around(f1_score(y_test, tree_prediction)[0], 3)
        macro_f1_score = np.around(f1_score(y_test, tree_prediction)[1], 3)
        table = np.stack((letters, tree_precision, tree_recall, tree_f1_score), 1)
        metrics = np.array([["Label", "Precision", "Recall", "F1-score"]])
        table = np.concatenate((metrics, table))
        tree_accuracy = accuracy(y_test, tree_prediction)
        macros.append(
            np.array(
                [file, tree_accuracy, macro_precision, macro_recall, macro_f1_score]
            )
        )
        print(
            "\n\n Evaluation metrics for binary decision tree trained on " + file + ":"
        )
        print(table)

    print("\n\nMacros for each decision tree:")
    top_row = np.array([["Dataset", "Accuracy", "Precision", "Recall", "F1-score"]])
    table = np.stack((macros[0], macros[1], macros[2]), 0)
    table = np.concatenate((top_row, table))
    print(table)

    print("\n\n\nClassifier Accuracy in 10-Fold Cross-Validation:")
    decisionTreeClassifier = DecisionTreeClassifier()
    n_splits = 10
    accuracies, fitted_classifiers = train_test_k_fold(
        decisionTreeClassifier, x_full, y_full, n_splits
    )
    for i in range(10):
        print(accuracies[i])

    print("Average accuracy: {}".format(np.mean(accuracies)))
    print("Standard deviation: {}".format(np.std(accuracies)))

    print(
        "\n\n-------------Evaluation of combining predictions from 10 decision trees generated by cross validation-------------\n\n"
    )
    newDecisionTreeClassifier = DecisionTreeClassifier()
    predictions = cross_validation_prediction(
        newDecisionTreeClassifier, x_full, y_full, x_test, n_splits
    )
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, predictions))

    macros = []
    letters = np.array(classes)
    tree_precision = np.around(precision(y_test, predictions)[0], 3)
    macro_precision = np.around(precision(y_test, predictions)[1], 3)
    tree_recall = np.around(recall(y_test, predictions)[0], 3)
    macro_recall = np.around(recall(y_test, predictions)[1], 3)
    tree_f1_score = np.around(f1_score(y_test, predictions)[0], 3)
    macro_f1_score = np.around(f1_score(y_test, predictions)[1], 3)
    table = np.stack((letters, tree_precision, tree_recall, tree_f1_score), 1)
    metrics = np.array([["Label", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((metrics, table))
    tree_accuracy = accuracy(y_test, predictions)
    macros.append(
        np.array([tree_accuracy, macro_precision, macro_recall, macro_f1_score])
    )
    print("\n\n Evaluation metrics for cross validation predictions:")
    print(table)

    print(
        "\n\n-------------Implementing A 4-Branch Multiway Decision Tree--------------\n\n"
    )
    print("\nTraining multiway classifier...\n")
    four_branch = MultiwayDecisionTreeClassifier(max_branches=4)
    four_branch.fit(x_full, y_full)

    print("\nMultiway classifier trained!\n")

    print(
        "\n\n-------------4 Branch Multiway Tree Evaluation Metrics--------------\n\n"
    )

    print("\nConfusion Matrix:\n")

    predictions = four_branch.predict(x_test)
    print(confusion_matrix(y_test, predictions))

    macros = []
    letters = np.array(classes)
    tree_precision = np.around(precision(y_test, predictions)[0], 3)
    macro_precision = np.around(precision(y_test, predictions)[1], 3)
    tree_recall = np.around(recall(y_test, predictions)[0], 3)
    macro_recall = np.around(recall(y_test, predictions)[1], 3)
    tree_f1_score = np.around(f1_score(y_test, predictions)[0], 3)
    macro_f1_score = np.around(f1_score(y_test, predictions)[1], 3)
    table = np.stack((letters, tree_precision, tree_recall, tree_f1_score), 1)
    metrics = np.array([["Label", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((metrics, table))
    tree_accuracy = accuracy(y_test, predictions)
    macros.append(
        np.array([tree_accuracy, macro_precision, macro_recall, macro_f1_score])
    )
    print("\n\n Evaluation metrics for a 4 branch multiway tree:")
    print(table)

    print("\n\nMacros for multiway decision tree trained on test.txt:")
    top_row = np.array([["Accuracy", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((top_row, macros))
    print(table)

    print(
        "\n\n-------------Implementing An Unconstrained Multiway Decision Tree--------------\n\n"
    )
    print("\nTraining multiway classifier...\n")
    unconstrained = MultiwayDecisionTreeClassifier()
    unconstrained.fit(x_full, y_full)

    print("\nMultiway classifier trained!\n")

    print(
        "\n\n-------------Unconstrained Multiway Tree Evaluation Metrics--------------\n\n"
    )

    print("\nConfusion Matrix:\n")

    predictions = unconstrained.predict(x_test)
    print(confusion_matrix(y_test, predictions))

    macros = []
    letters = np.array(classes)
    tree_precision = np.around(precision(y_test, predictions)[0], 3)
    macro_precision = np.around(precision(y_test, predictions)[1], 3)
    tree_recall = np.around(recall(y_test, predictions)[0], 3)
    macro_recall = np.around(recall(y_test, predictions)[1], 3)
    tree_f1_score = np.around(f1_score(y_test, predictions)[0], 3)
    macro_f1_score = np.around(f1_score(y_test, predictions)[1], 3)
    table = np.stack((letters, tree_precision, tree_recall, tree_f1_score), 1)
    metrics = np.array([["Label", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((metrics, table))
    tree_accuracy = accuracy(y_test, predictions)
    macros.append(
        np.array([tree_accuracy, macro_precision, macro_recall, macro_f1_score])
    )
    print(
        "\n\n Evaluation metrics for an unconstrained decision tree trained on train_full.txt:"
    )
    print(table)

    print("\n\nMacros for multiway decision tree:")
    top_row = np.array([["Accuracy", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((top_row, macros))
    print(table)

    print("\n\n-------------Pruning A Binary Decision Tree--------------\n\n")

    indexes = [0, 2]
    for i in indexes:
        tree = bin_trees[i]
        file = files[i]
        print("\n\n Pruning a binary tree trained on " + file + ":\n")
        prior_node_count = tree.node_count
        predictions_v = tree.predict(x_val)
        prior_acc_v = accuracy(y_val, predictions_v)
        predictions_t = tree.predict(x_test)
        prior_acc_t = accuracy(y_test, predictions_t)
        tree.prune(x_val, y_val)
        post_node_count = tree.node_count
        predictions_v = tree.predict(x_val)
        post_acc_v = accuracy(y_val, predictions_v)
        predictions_t = tree.predict(x_test)
        post_acc_t = accuracy(y_test, predictions_t)
        print("     Prior accuracy on validation set:", prior_acc_v)
        print("     Prior accuracy on test set:", prior_acc_t)
        print("     Prior node count:", prior_node_count)
        print("\n     Post accuracy on validation set:", post_acc_v)
        print("     Post accuracy on test set:", post_acc_t)
        print("     Post node count:", post_node_count)

    print(
        "\n\n-------------Pruning A four Way Multiway Decision Tree--------------\n\n"
    )

    tree = four_branch
    prior_node_count = tree.node_count
    predictions_v = tree.predict(x_val)
    prior_acc_v = accuracy(y_val, predictions_v)
    predictions_t = tree.predict(x_test)
    prior_acc_t = accuracy(y_test, predictions_t)
    tree.prune(x_val, y_val)
    post_node_count = tree.node_count
    predictions_v = tree.predict(x_val)
    post_acc_v = accuracy(y_val, predictions_v)
    predictions_t = tree.predict(x_test)
    post_acc_t = accuracy(y_test, predictions_t)
    print("     Prior accuracy on validation set:", prior_acc_v)
    print("     Prior accuracy on test set:", prior_acc_t)
    print("     Prior node count:", prior_node_count)
    print("\n     Post accuracy on validation set:", post_acc_v)
    print("     Post accuracy on test set:", post_acc_t)
    print("     Post node count:", post_node_count)

    print(
        "\n\n-------------Pruning An Unconstrained Multiway Decision Tree--------------\n\n"
    )

    tree = unconstrained
    prior_node_count = tree.node_count
    predictions_v = tree.predict(x_val)
    prior_acc_v = accuracy(y_val, predictions_v)
    predictions_t = tree.predict(x_test)
    prior_acc_t = accuracy(y_test, predictions_t)
    tree.prune(x_val, y_val)
    post_node_count = tree.node_count
    predictions_v = tree.predict(x_val)
    post_acc_v = accuracy(y_val, predictions_v)
    predictions_t = tree.predict(x_test)
    post_acc_t = accuracy(y_test, predictions_t)
    print("     Prior accuracy on validation set:", prior_acc_v)
    print("     Prior accuracy on test set:", prior_acc_t)
    print("     Prior node count:", prior_node_count)
    print("\n     Post accuracy on validation set:", post_acc_v)
    print("     Post accuracy on test set:", post_acc_t)
    print("     Post node count:", post_node_count)

    print("\n\n-------------Implementing A Random Decision Forest--------------\n\n")
    print("\nTraining random decision forest...\n")
    random_forest = RandomForest(50, DecisionTreeClassifier, 3)
    random_forest.fit(x_full, y_full, x_val, y_val)


    print("\nRandom decision forest trained!\n")

    print(
        "\n\n-------------Random Decision Forest Evaluation Metrics--------------\n\n"
    )

    print("\nConfusion Matrix:\n")

    predictions = random_forest.predict(x_test)
    print(confusion_matrix(y_test, predictions))

    macros = []
    letters = np.array(classes)
    tree_precision = np.around(precision(y_test, predictions)[0], 3)
    macro_precision = np.around(precision(y_test, predictions)[1], 3)
    tree_recall = np.around(recall(y_test, predictions)[0], 3)
    macro_recall = np.around(recall(y_test, predictions)[1], 3)
    tree_f1_score = np.around(f1_score(y_test, predictions)[0], 3)
    macro_f1_score = np.around(f1_score(y_test, predictions)[1], 3)
    table = np.stack((letters, tree_precision, tree_recall, tree_f1_score), 1)
    metrics = np.array([["Label", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((metrics, table))
    tree_accuracy = accuracy(y_test, predictions)
    macros.append(
        np.array([tree_accuracy, macro_precision, macro_recall, macro_f1_score])
    )
    print("\n\n Evaluation metrics for random decision forest:")
    print(table)

    print("\n\nMacros for random decision forest:")
    top_row = np.array([["Accuracy", "Precision", "Recall", "F1-score"]])
    table = np.concatenate((top_row, macros))
    print(table)
