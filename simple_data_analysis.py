"""
File:           simple_data_analysis.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  30/01/2022
Last Edit Date: 31/02/2022
Last Edit By:   Ted Jenks

Summary of File:

        Contains simple analytics of the training data.
"""

from read_data import read_data
import numpy as np
import matplotlib.pyplot as plt

files = ["train_full.txt", "train_sub.txt", "train_noisy.txt"]

x_full, y_full = read_data("data/" + files[0])
x_sub, y_sub = read_data("data/" + files[1])
x_noisy, y_noisy = read_data("data/" + files[2])

x_array = [x_full, x_sub, x_noisy]
y_array = [y_full, y_sub, y_noisy]


def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(
            _X - width / 2.0 + i / float(n) * width,
            vals[i],
            width=width / float(n),
            align="edge",
            label=files[i],
        )
    plt.xticks(_X, X)


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
    print("      Classes :\n\n", class_proportion)


subcategorybar(classes, [proportions_plot[0], proportions_plot[1], proportions_plot[2]])
plt.xlabel("Catagory")
plt.ylabel("Proportion")
plt.title("Distribution of Classes in Data")
plt.legend()
plt.show()

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

attributes = np.arange(1, 17)
subcategorybar(attributes, [range_plot[0], range_plot[1], range_plot[2]])
plt.xlabel("Attribute")
plt.ylabel("Range")
plt.title("Range of values in attributes")
plt.legend()
plt.show()


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
