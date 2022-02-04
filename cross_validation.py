import numpy as np

def k_fold_split(n_splits, n_instances):
    # generate random permutation of [0...n_instances] (where n_instances == the total number
    # of data points in your data)
    shuffled_indices = np.random.default_rng().permutation(n_instances)

    # now I have one permutation of n_instances length which I need to partition
    # into n_splits. Note: We use array_split instead of .split() because we may need
    # our permutation into partitions of unequal length (not supported by split())
    split_indices = np.array_split(shuffled_indices, n_splits)
    
    return split_indices

def train_test_k_fold(decision_tree_classifier, train_attr, train_labels, val_attr, val_labels, test_attr, test_labels, n_splits):
    full_attr = np.concatenate((train_attr, val_attr, test_attr))
    full_labels = np.concatenate((train_labels, val_labels, test_labels))

    # Number of total instances to split
    n_instances = np.shape(full_attr)[0]

    split_indices = k_fold_split(n_splits, n_instances)
    split_indices = np.array(split_indices)

    folds = []
    for i in range(n_splits):
        test_indices = split_indices[i]
        v = (i + 1) % n_splits
        val_indices = split_indices[v]
        train_indices = np.delete(split_indices, [i, v], 0).flatten()

        index_set = [train_indices, val_indices, test_indices]
        
        folds.append(index_set)

    # Currently, folds[] has 10 rows (as per n_splits argument) and 3 columns;
    # first column is train_indices, 2nd is val_indices, and 3rd is test_indices
    accuracies = []
    


    for i in range(n_splits):
        x_train = full_attr[folds[i][0]]
        y_train = full_labels[folds[i][0]]

        decision_tree_classifier.fit(x_train, y_train)

        x_val = full_attr[folds[i][1]]
        y_val = full_labels[folds[i][1]]

        decision_tree_classifier.prune(x_val, y_val)

        x_test = full_attr[folds[i][2]]
        y_test = full_labels[folds[i][2]]

        predictions = decision_tree_classifier.predict(x_test)

        accuracy = (np.count_nonzero(predictions == y_test)) / y_test.size
        accuracies.append(accuracy)

    return (np.mean(accuracies), np.std(accuracies))