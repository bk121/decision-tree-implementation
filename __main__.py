"""
File:           __main__.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Summary of File:

        Contains main file to run classifier(s).
"""
import numpy as np
from cross_validation_prediction import cross_validation_prediction
from torch import cross
from classification import DecisionTreeClassifier
from read_data import read_data
from evaluation_metrics import accuracy, confusion_matrix, precision, recall, f1_score
import pickle
import matplotlib.pyplot as plt
import train_classifiers
from cross_validation import train_test_k_fold




if __name__ == "__main__":

    print("\n\n-------------Examining the dataset--------------\n\n")

    files = ["train_full.txt", "train_sub.txt", "train_noisy.txt"]

    x_full, y_full = read_data("data/" + files[0])
    x_sub, y_sub = read_data("data/" + files[1])
    x_noisy, y_noisy = read_data("data/" + files[2])
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
    tree_names=["bin_full", "bin_sub", "bin_noisy"]
    class_files=["trained_classifiers/"+tree_name+".pickle" for tree_name in tree_names]
    data_files=["data/"+file for file in files]
    train_classifiers.binary_train(tree_names, class_files, data_files)
    print("\nBinary classifiers trained!\n")


    print("\n\n-------------Evaluation Metrics--------------\n\n")

    print("\nConfusion Matrices:\n")
    trees=[]
    for class_file, data_file, tree in zip(class_files, data_files, tree_names):
        print("\nConfusion matrix for Binary decision tree (Trained on "+data_file+"):")
        classifier=open(class_file, "rb")
        tree=pickle.load(classifier)
        trees.append(tree)
        classifier.close()
        print(confusion_matrix(y_test, tree.predict(x_test)))

    print("\n\n\nMetrics per label for each decision tree:")
    macros=[]
    for tree, file in zip(trees, files):
        tree_prediction=tree.predict(x_test)
        letters=np.array(classes)
        tree_precision=np.around(precision(y_test, tree_prediction)[0], 3)
        macro_precision=np.around(precision(y_test, tree_prediction)[1], 3)
        tree_recall=np.around(recall(y_test, tree_prediction)[0], 3)
        macro_recall=np.around(recall(y_test, tree_prediction)[1], 3)
        tree_f1_score=np.around(f1_score(y_test, tree_prediction)[0], 3)
        macro_f1_score=np.around(f1_score(y_test, tree_prediction)[1], 3)
        table=np.stack((letters, tree_precision, tree_recall, tree_f1_score),1)
        metrics=np.array([["Label", "Precision", "Recall", "F1-score"]])
        table=np.concatenate((metrics, table))
        tree_accuracy=accuracy(y_test, tree_prediction)
        macros.append(np.array([file, tree_accuracy, macro_precision, macro_recall, macro_f1_score]))
        print("\n\n Evaluation metrics for binary decision tree trained on "+file+":")
        print(table)
    
    print("\n\nMacros for each decision tree:")
    top_row=np.array([["Dataset", "Accuracy", "Precision", "Recall", "F1-score"]])
    table=np.stack((macros[0], macros[1], macros[2]),0)
    table=np.concatenate((top_row, table))
    print(table)
    
    print("\n\n\nClassifier Accuracy in 10-Fold Cross-Validation:")
    decisionTreeClassifier = DecisionTreeClassifier()
    n_splits = 10
    accuracies, fitted_classifiers = train_test_k_fold(decisionTreeClassifier, x_full, y_full, n_splits)
    for i in range(10):
        print(accuracies[i])

    print("Average accuracy: {}".format(np.mean(accuracies)))
    print("Average standard deviation: {}".format(np.std(accuracies)))

    print("\n\n\nAccuracy of combining predictions from 10 decision trees generated by cross validation:")
    newDecisionTreeClassifier = DecisionTreeClassifier()
    cross_validated_predictions = cross_validation_prediction(newDecisionTreeClassifier, x_full, y_full, x_test, n_splits)
    print(accuracy(y_test, cross_validated_predictions))


# files = {
#     "binary": "binary",
#     "pruned binary": "binary-pruned",
#     "max two branch multiway": "mw-two-way",
#     "pruned max two branch multiway": "mw-two-way-pruned",
#     "max three branch multiway": "mw-three-way",
#     "pruned max three branch multiway": "mw-three-way-pruned",
#     "max four branch multiway": "mw-four-way",
#     "pruned max four branch multiway": "mw-four-way-pruned",
#     "no limit multiway": "mw-inf-way",
#     "pruned no limit multiway": "mw-inf-way-pruned",
# }

    # x_test, y_test = read_data("data/test.txt")
    # for key in files:
    #     classifier_f = open(
    #         "trained_classifiers/" + files[key] + ".pickle",
    #         "rb",
    #     )
    #     classifier = pickle.load(classifier_f)
    #     classifier_f.close()
    #     print(
    #         "\n\n-------------------------------------------"
    #         + "----------------------------------------------------------\n"
    #     )
    #     print("\nAnalysis for", key, "decision tree:\n\n")
    #     predictions = classifier.predict(x_test)
    #     print("\nAccuracy:\n", accuracy(y_test, predictions))
    #     print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    #     print("\nPrecision:\n", precision(y_test, predictions))
    #     print("\nRecall:\n", recall(y_test, predictions))
    #     print("\nF1_Score:\n", f1_score(y_test, predictions))
    #     print("\nNode Count:\n", classifier.node_count)
