from sklearn.model_selection import KFold

def KFold_cross_validation_split(features, labels, n_splits):
    """
    KFold Cross Validation split

    Splits the data:
        Let K be the number of folds => 
        Training data = (100% - K%)
        Test data = K%

    params:
        <NDarray> Features
        <NDarray> Labels
        <int> n_splits

    returns:
        <NDarray> x_train = Feature train set
        <NDarray> x_test = Feature test set
        <NDarray> y_train = Label train set
        <NDarray> y_test = Label test set
    """

    kfold = KFold(n_splits = n_splits, shuffle = True)
    for train_idx, test_idx in kfold.split(features):
        x_train, x_test = x[train_idx], x[valid_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test