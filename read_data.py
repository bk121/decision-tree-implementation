"""
File:           read_data.py
Author:         Jonas Birk, Ted Jenks, Tom Mitcheson, Ben Kirwan
Creation Date:  30/01/2022
Last Edit Date: 31/02/2022
Last Edit By:   Ted Jenks

Public Functions:   read_data(filepath)

Summary of File:

        Contains data reader.
"""

import numpy as np


def read_data(filepath):
    """Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ) containing letters
                   in A-E representing the class
    """
    x = []
    y = []
    with open(filepath) as file:  # ensure file is closed
        for line in file:
            if line.strip() != "":  # ignore empty lines
                row = line.strip().split(",")  # delimit by comma, strip ws
                x.append(list(map(int, row[:-1])))
                y.append(row[-1])

    x = np.array(x)
    y = np.array(y)
    return (x, y)
