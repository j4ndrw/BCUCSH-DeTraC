import os

import cv2

import tensorflow as tf
import numpy as np

from tqdm import tqdm


def preprocess_single_image(img, width, height, framework: str, imagenet: bool = False):
    assert framework == "tf" or framework == "torch"

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    if imagenet == True:
        if framework == "torch":
            img = tf.keras.applications.imagenet_utils.preprocess_input(
                img, mode="torch")
        else:
            img = tf.keras.applications.imagenet_utils.preprocess_input(
                img, mode="tf")
    return img


def preprocess_images(dataset_path, width, height, num_classes, framework: str, imagenet: bool = False):
    """
    Preprocessing of dataset before training.

    params:
        <string> dataset_path = Path where raw data is located
        <int> width = Width of image
        <int> height = Height of image
        <string> framework = Choice of framework

    returns:
        <NDarray> Features
        <NDarray> Labels
    """

    assert framework == "tf" or framework == "torch"

    features = []
    labels = []

    identity_array = list(np.eye(num_classes, dtype=np.int32))

    class_names = []
    for folder in os.listdir(dataset_path):
        assert os.path.isdir(os.path.join(dataset_path, folder))
        class_names.append(folder)

    for folder in os.listdir(dataset_path):
        file_progress_bar = tqdm(os.listdir(
            os.path.join(dataset_path, folder)))
        for filename in file_progress_bar:
            if filename.lower().endswith("png") or filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):
                gray_img = cv2.imread(os.path.join(
                    dataset_path, folder, filename))
                color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(color_img, (width, height))
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                if imagenet == True:
                    if framework == "torch":
                        img = tf.keras.applications.imagenet_utils.preprocess_input(
                            img, mode="torch")
                    else:
                        img = tf.keras.applications.imagenet_utils.preprocess_input(
                            img, mode="tf")
                features.append(img)
                labels.append(identity_array[class_names.index(folder)])

                file_progress_bar.set_description(
                    f"Loading images from directory {folder}")

    features = np.vstack(features)
    labels = np.array(labels)

    return class_names, features, labels
