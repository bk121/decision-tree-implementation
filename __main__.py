import binary_classification
import multiway_classification
from multiway_classification import MultiwayDecisionTreeClassifier
from read_data import read_data
from evaluation_metrics import accuracy, confusion_matrix, precision, recall, f1_score
import pickle

files = {
    "binary": "binary",
    "pruned binary": "binary-pruned",
    "max two branch multiway": "mw-two-way",
    "pruned max two branch multiway": "mw-two-way-pruned",
    "max three branch multiway": "mw-three-way",
    "pruned max three branch multiway": "mw-three-way-pruned",
    "max four branch multiway": "mw-four-way",
    "pruned max four branch multiway": "mw-four-way-pruned",
    "no limit multiway": "mw-inf-way",
    "pruned no limit multiway": "mw-inf-way-pruned",
}


if __name__ == "__main__":
    x_test, y_test = read_data("data/test.txt")
    for key in files:
        classifier_f = open(
            "trained_classifiers/" + files[key] + ".pickle",
            "rb",
        )
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        print(
            "\n\n-------------------------------------------"
            + "----------------------------------------------------------\n"
        )
        print("\nAnalysis for", key, "decision tree:\n\n")
        predictions = classifier.predict(x_test)
        print("\nAccuracy:\n", accuracy(y_test, predictions))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
        print("\nPrecision:\n", precision(y_test, predictions))
        print("\nRecall:\n", recall(y_test, predictions))
        print("\nF1_Score:\n", f1_score(y_test, predictions))
        print("\nNode Count:\n", classifier.node_count)
