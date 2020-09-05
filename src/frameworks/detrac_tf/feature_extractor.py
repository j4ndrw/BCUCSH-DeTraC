import tensorflow as tf

from tensorflow.keras.applications import VGG16

import numpy as np

from tools.preprocessing import preprocess_images
from tools.kfold import KFold_cross_validation_split
from tools.extraction_and_metrics import extract_features, compute_confusion_matrix

from .network import Net

import os
import cv2

# Feature extractor training
def train_feature_extractor(
    initial_dataset_path,
    extracted_features_path,
    epochs,
    batch_size,
    num_classes,
    folds,
    model_dir
):
    """
    Feature extractor training.

    params:
     <string> initial_dataset_path
     <string> extracted_features_path
     <int> epochs
     <int> batch_size
     <int> num_classes
     <int> folds: Number of folds for KFold cross validation 
     <string> model_dir: Model's location
    """

    # Preprocess images, returning the classes, features and labels
    class_names, x, y = preprocess_images(
        initial_dataset_path, 224, 224, num_classes, framework="tf")

    # Split data
    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(
        x, y, folds)

    # Normalize
    X_train /= 255
    X_test /= 255

    # Instantiate model
    net = Net(
        pretrained_model=VGG16(
            input_shape=(224, 224, 3),
            include_top=True
        ),
        num_classes=num_classes,
        mode="feature_extractor",
        class_names=class_names,
        model_dir=model_dir
    )

    # Train model
    net.fit(
        x_train=X_train,
        y_train=Y_train,
        x_test=X_test,
        y_test=Y_test,
        epochs=epochs,
        batch_size=batch_size,
        resume=False
    )

    # Extract features
    for class_name in class_names:
        extracted_features = extract_features(
            initial_dataset_path, class_name, 224, 224, net, framework="tf")
        np.save(os.path.join(extracted_features_path,
                             f"{class_name}.npy"), extracted_features)

    # Confusion matrics
    compute_confusion_matrix(y_true=Y_test, y_pred=net.infer(
        X_test, use_labels=False), framework="tf", mode="feature_extractor", num_classes = num_classes)
