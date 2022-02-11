from multiway_classification import MultiwayDecisionTreeClassifier
from classification import DecisionTreeClassifier
from read_data import read_data
import pickle

x_test, y_test = read_data("data/test.txt")

def binary_train(classifiers, class_files, data_files):

    for classifier, data_file, class_file in zip(classifiers, data_files, class_files):
        x,y=read_data(data_file)
        classifier=DecisionTreeClassifier()
        classifier.fit(x,y)
        save_cl=open(class_file, "wb")
        pickle.dump(classifier, save_cl)
        save_cl.close()

