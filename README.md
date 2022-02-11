## Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.


### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``improvement.py``

	* Contains the skeleton code for the ``train_and_predict()`` function (Task 4.1).
Complete this function as an interface to your new/improved decision tree classifier.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods/functions defined in ``classification.py`` and ``improvement.py``.


### Instructions

In addition to the above, we have split our functions and classes into separate files for clarity,
and imported them to other files as needed.
To make it clear how each function or class is being used to generate the results in our report,
we created a file ``__main__.py`` which produces and prints out the results included in the report. 
In some cases, there will be small variations as a result of introducing randomness. For example,
the indices used for splitting training and test data in cross-validation are generated randomly,
and so the 10-fold cross validation results will differ slightly every time they are generated.



