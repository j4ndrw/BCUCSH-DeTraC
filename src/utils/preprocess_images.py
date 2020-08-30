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

    identity_array = np.eye(num_classes, dtype = np.int32)

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
            labels.append(identity_array[class_names.index(folder)])

    return np.vstack(features), np.array(labels)