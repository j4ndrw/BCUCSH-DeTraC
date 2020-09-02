import os

import cv2

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from .multiclass_confusion_matrix import multiclass_confusion_matrix

from tqdm import tqdm


def extract_features(initial_dataset_path, class_name, width, height, net, framework):
    assert framework == "tf" or framework == "torch"

    features = []

    for filename in tqdm(os.listdir(os.path.join(initial_dataset_path, class_name))):
        gray_img = cv2.imread(os.path.join(
            initial_dataset_path, class_name, filename))
        color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(color_img, (width, height))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        if framework == "torch":
            img = tf.keras.applications.imagenet_utils.preprocess_input(
                img, mode="torch")
        else:
            img = tf.keras.applications.imagenet_utils.preprocess_input(
                img, mode="tf")
        features.append(img)

    features = np.vstack(features)
    return net.infer_using_pretrained_layers_without_last(features)


def compose_classes(cmat, block_size: tuple):
    sizes = list(tuple(np.array(cmat.shape) // block_size) + block_size)
    for i in range(len(sizes)):
        if (i + 1) == len(sizes) - 1:
            break
        if i % 2 != 0:
            temp = sizes[i]
            sizes[i] = sizes[i + 1]
            sizes[i + 1] = temp

    reshaped_matrix = cmat.reshape(sizes)
    composed = reshaped_matrix.sum(axis=(1, 3))
    return composed


def compute_confusion_matrix(y_true, y_pred, framework: str, mode: str, num_classes: int):
    assert framework == "tf" or framework == "torch"
    assert mode == "feature_extractor" or mode == "feature_composer"

    cmat = confusion_matrix(y_true.argmax(
        axis=1), y_pred.argmax(axis=1), normalize="all")
    
    if mode == "feature_composer":
        cmat = compose_classes(cmat, (2, 2))
        
    print(cmat)

    acc, sn, sp = multiclass_confusion_matrix(cmat, num_classes)

    output = f"ACCURACY = {acc}\nSENSITIVITY = {sn}\nSPECIFICITY = {sp}"
#     if framework == "torch":
#         log_dir = "../../../models/torch/logs"
#     else:
#         log_dir = "../../../models/tf/logs"
#     with open(os.path.join(log_dir, "metrics_log.txt"), 'w') as f:
#         f.write(output)
    print(output)
