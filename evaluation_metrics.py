"""
File:           evaluation_metrics.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  31/01/2022
Last Edit Date: 10/02/2022
Last Edit By:   Ted Jenks

Public Functions:   confusion_matrix(gold_labels, prediction_labels, class_labels),
                    accuracy(gold_labels, prediction_labels),
                    precision(gold_labels, prediction_labels),
                    recall(gold_labels, prediction_labels),
                    f1_score(gold_labels, prediction_labels)

Summary of File:

        Contains evaluation metrics for the decision trees.
"""

import numpy as np


def confusion_matrix(gold_labels, prediction_labels, class_labels=None):
    """
    Create a confusion matrix for a classifier given the true dta and
    predictions

    Args:
        gold_labels (np.ndarray): Array of true data labels.
        prediction_labels ([type]): Array of predicted data labels.
        class_labels (np.ndarry, optional): Array of labels. Defaults to None.

    Returns:
        np.ndarray: Confusion matrix
    """

    # if no class labels give, default to the union of prediction and gold standard
    if not class_labels:
        class_labels = np.unique(np.concatenate((gold_labels, prediction_labels)))

    # create empty confusion matrix
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
    np.sort(class_labels)

    enum_letter_dict = dict(
        zip(np.unique(class_labels), range(len(np.unique(class_labels))))
    )
    # print(enum_letter_dict)

    # change labels into their numerical counterparts in dict
    num_gold_labels = [enum_letter_dict[x] for x in gold_labels]
    num_prediction_labels = [enum_letter_dict[x] for x in prediction_labels]

    # compute how many instances of each class
    for i in range(len(num_gold_labels)):
        confusion[num_gold_labels[i]][num_prediction_labels[i]] += 1

    return confusion


def accuracy(gold_labels, prediction_labels):
    """
    Calculate accuracy of the predictions.

    Args:
        gold_labels (np.ndarray): Array of true data labels.
        prediction_labels ([type]): Array of predicted data labels.

    Returns:
        flt: Accuracy of predictions
    """

    assert len(gold_labels) == len(prediction_labels)

    try:
        return np.sum(gold_labels == prediction_labels) / len(gold_labels)
    except ZeroDivisionError:
        return 0


def precision(gold_labels, prediction_labels):
    """
    Calculate the precision(s) of the predictions

    Args:
        gold_labels (np.ndarray): Array of true data labels.
        prediction_labels ([type]): Array of predicted data labels.

    Returns:
        np.ndarray: The precision(s) of the predictions.
    """

    confusion = confusion_matrix(gold_labels, prediction_labels)
    class_labels = np.unique(np.concatenate((gold_labels, prediction_labels)))

    # create a precision vector (or 1D array) for each class' precision value
    p = np.zeros([len(class_labels),]).reshape(
        len(class_labels),
    )
    for i in range(len(np.unique(gold_labels))):
        p[i,] = confusion[i][
            i
        ] / (confusion[:, i].sum())

    # macro-averaged precision
    macro_p = float(p.sum() / len(p))

    return (p, macro_p)


def recall(gold_labels, prediction_labels):
    """
    Evaluate recall(s) of the predictions.

    Args:
        gold_labels (np.ndarray): Array of true data labels.
        prediction_labels ([type]): Array of predicted data labels.

    Returns:
        np.ndarray: The recall(s) of the predictions
    """

    confusion = confusion_matrix(gold_labels, prediction_labels)
    class_labels = np.unique(np.concatenate((gold_labels, prediction_labels)))

    # create a recall vector (or 1D array) for each class' recall value
    r = np.zeros([len(class_labels),]).reshape(
        len(class_labels),
    )
    for i in range(len(np.unique(gold_labels))):
        r[i,] = confusion[i][
            i
        ] / (confusion[i, :].sum())

    # macro-averaged recall
    macro_r = r.sum() / len(r)

    return (r, macro_r)


def f1_score(gold_labels, prediction_labels):
    """
    Evaluate F1-score of the data.

    Args:
        gold_labels (np.ndarray): Array of true data labels.
        prediction_labels ([type]): Array of predicted data labels.

    Returns:
        np.ndarray: The F1-score(s) of the predictions
    """

    # Get recalls and precisions
    (precisions, macro_p) = precision(gold_labels, prediction_labels)
    (recalls, macro_r) = recall(gold_labels, prediction_labels)

    assert len(precisions) == len(recalls)

    # Array to hold results
    f = np.zeros((len(precisions),))
    for i in range(len(np.unique(gold_labels))):
        # Get results
        if (precisions[i] + recalls[i]) > 0:
            f[i,] = (
                2 * precisions[i] * recalls[i]
            ) / (precisions[i] + recalls[i])

    # Get macro value
    macro_f = f.sum() / len(f)

    return (f, macro_f)
