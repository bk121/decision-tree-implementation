import numpy as np

def confusion_matrix(gold_labels, prediction_labels, class_labels=None):

    # if no class labels give, default to the union of prediction and gold standard
    if not class_labels:
        class_labels = np.unique(np.concatenate((gold_labels, prediction_labels)))

    # create empty confusion matrix
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
    np.sort(class_labels)

    enum_letter_dict = dict(zip(np.unique(class_labels), range(len(np.unique(class_labels)))))
    #print(enum_letter_dict)

    # change labels into their numerical counterparts in dict
    num_gold_labels = [enum_letter_dict[x] for x in gold_labels]
    num_prediction_labels = [enum_letter_dict[x] for x in prediction_labels]

    # compute how many instances of each class
    for i in range(len(num_gold_labels)):
        confusion[num_gold_labels[i]][num_prediction_labels[i]] += 1

    return confusion

def accuracy(gold_labels, prediction_labels):

    assert len(gold_labels) == len(prediction_labels)

    try:
        return np.sum(gold_labels == prediction_labels)/len(gold_labels)
    except ZeroDivisionError:
        return 0

def precision(gold_labels, prediction_labels):
    
    confusion = confusion_matrix(gold_labels, prediction_labels)
    class_labels = np.unique(np.concatenate((gold_labels, prediction_labels)))

    # create a precision vector (or 1D array) for each class' precision value
    p = np.zeros([len(class_labels),]).reshape(len(class_labels),)
    for i in range(len(np.unique(gold_labels))):
        p[i,] = confusion[i][i]/(confusion[:,i].sum())

    # macro-averaged precision
    macro_p = float(p.sum()/len(p))

    return(p, macro_p)

def recall(gold_labels, prediction_labels):

    confusion = confusion_matrix(gold_labels, prediction_labels)
    class_labels = np.unique(np.concatenate((gold_labels, prediction_labels)))

    # create a recall vector (or 1D array) for each class' recall value
    r = np.zeros([len(class_labels),]).reshape(len(class_labels),)
    for i in range(len(np.unique(gold_labels))):
        r[i,] = confusion[i][i]/(confusion[i,:].sum())

    # macro-averaged recall
    macro_r = r.sum()/len(r)

    return(r, macro_r)

def f1_score(gold_labels, prediction_labels):

    (precisions, macro_p) = precision(gold_labels, prediction_labels)
    (recalls, macro_r) = recall(gold_labels, prediction_labels)

    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for i in range(len(np.unique(gold_labels))):
        if(precisions[i] + recalls[i]) > 0:
            f[i,] = (2 * precisions[i] * recalls[i])/(precisions[i] + recalls[i])

    macro_f = f.sum()/len(f)

    return(f, macro_f)
    