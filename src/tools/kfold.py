import numpy as np
from sklearn.model_selection import KFold

def KFold_cross_validation_split(
    features: np.ndarray, 
    labels: np.ndarray, 
    n_splits: int
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    KFold Cross Validation split

    Splits the data:
        Let K be the number of folds => 
        Training data = (100% - K%)
        Test data = K%

    params:
        <array> Features
        <array> Labels
        <int> n_splits

    returns:
        <array> x_train = Feature train set
        <array> x_test = Feature test set
        <array> y_train = Label train set
        <array> y_test = Label test set
    """

    kfold = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in kfold.split(features):
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

    return x_train, x_test, y_train, y_test
