from random import sample
import numpy as np

x = [[2, 3], [3, 4]]

print(np.shape(x)[1])
number_of_trees = 10
num_of_attributes = 10
num_of_attributes_in_each_tree = 2

attributes_subset = np.empty(
    (number_of_trees, num_of_attributes_in_each_tree),
)
attributes_subset = np.array(
    [
        sample((range(num_of_attributes)), num_of_attributes_in_each_tree)
        for i in range(number_of_trees)
    ]
)


print(attributes_subset)
