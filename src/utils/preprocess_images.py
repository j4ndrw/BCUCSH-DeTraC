import os

import cv2

import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

from tqdm import tqdm

def initial_preprocessing(dataset_path, (width, height), num_classes):
    """
    Preprocessing of dataset before training.

    params:
        <string> dataset_path = Path where raw data is located
        <tuple> (width, height) = Size of image

    returns:
        <NDarray> Features
        <NDarray> Labels
    """
    
    features = []
    labels = []

    identity_NDarray = np.eye(num_classes, dtype = np.int32)

    class_names = []
    for folder in os.listdir(dataset_path):
        assert os.path.isdir(folder)
        class_names.append(folder)

    for folder in tqdm(os.listdir(dataset)):
        for filename in os.listdir(folder):
            gray_img = cv2.imread(os.path.join(dataset, filename))
            color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(color_img, (width, height))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis = 0)
            features.append(img)
            labels.append(identity_NDarray[class_names.index(folder)])

    return np.vstack(features), np.array(labels)

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