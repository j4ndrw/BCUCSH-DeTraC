import tensorflow as tf

from tensorflow.keras.applications import VGG16

import numpy as np

from utils.preprocessing import preprocess_images
from utils.kfold import KFold_cross_validation_split
from utils.extraction_and_metrics import extract_features, compute_confusion_matrix

from .network import Net

import os
import cv2

def train_feature_composer(
        initial_dataset_path,
        extracted_features_path,
        epochs, 
        batch_size, 
        num_classes, 
        folds
    ):

    class_names, x, y = preprocess_images(initial_dataset_path, 224, 224, num_classes, framework = "tf")

    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(x, y, folds)

    X_train /= 255
    X_test /= 255

    net = Net(
        pretrained_model = VGG16(
            input_shape = (224, 224, 3),
            include_top = True
        ),
        num_classes = num_classes,
        mode = "feature_composer",
        class_names = class_names
    )

    net.fit(
        x_train = X_train,
        y_train = Y_train,
        x_test = X_test,
        y_test = Y_test,
        epochs = epochs,
        batch_size = batch_size, 
        resume = False
    )

    compute_confusion_matrix(y_true = Y_test, y_pred = net.infer(X_test), framework = "tf", mode = "feature_composer")

def infer(model_path, input_image):
    net = Net(
        pretrained_model = VGG16(
            input_shape = (224, 224, 3),
            include_top = True
        ),
        num_classes = num_classes,
        mode = "feature_composer",
        class_names = class_names
    )
    tf.keras.models.load_model(model_path, compile = False)
    assert input_image.lower().endswith("png") or input_image.lower().endswith("jpg") or input_image.lower().endswith("jpeg")
    img = preprocess_single_image(input_image, 224, 224, imagenet = True, framework = "tf")
    return net.infer(img)