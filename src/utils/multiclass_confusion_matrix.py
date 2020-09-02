import numpy as np


def multiclass_confusion_matrix(cmat, num_classes):
    """
    Multi-class confusion matrix.

    params:
        - <matrix> cmat = Confusion Matrix (from sklearn.metrics.confusion_matrix)
        - <int> num_classes = The number of classes to be used when tracking the true or false positives and true or false negatives respectively.

    returns:
        <float> Accuracy
        <float> Sensitivity
        <float> Specificity
    """

    acc = np.zeros(num_classes)
    sn = np.zeros(num_classes)
    sp = np.zeros(num_classes)

    for class_idx in range(num_classes):
        TP, TN, FP, FN = 0, 0, 0, 0

        # True positives are located in the matrix's primary diagonal.
        TP += cmat[class_idx][class_idx]

        for i in range(num_classes):
            if i == class_idx:
                for j in range(num_classes):
                    if j != i:
                        # False negatives are located under the matrix's primary diagonal.
                        FN += cmat[i][j]

                        # False positives are located above the matrix's primary diagonal.
                        FP += cmat[j][i]
            else:
                for j in range(num_classes):
                    if j != class_idx:
                        # True negatives are located anywhere but where the index of the iteration is.
                        TN += cmat[i][j]

        acc[class_idx] = (TP + TN) / (TP + TN + FP + FN)
        sn[class_idx] = TP / (TP + FN)
        sp[class_idx] = TN / (TN + FP)

    return np.mean(ACC_Class), np.mean(SN_Class), np.mean(SP_Class)
