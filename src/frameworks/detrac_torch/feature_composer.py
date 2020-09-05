import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

from tools.preprocessing import preprocess_images, preprocess_single_image
from tools.kfold import KFold_cross_validation_split
from tools.extraction_and_metrics import extract_features, compute_confusion_matrix

from .network import Net

import torchvision.models as models
import torch

import os
import cv2

# Feature composer training
def train_feature_composer(
    composed_dataset_path,
    epochs,
    batch_size,
    num_classes,
    folds,
    cuda,
    ckpt_dir
):
    """
    Feature extractor training.

    params:
     <string> composed_dataset_path
     <int> epochs
     <int> batch_size
     <int> num_classes
     <int> folds: Number of folds for KFold cross validation 
     <bool> cuda: Whether to use GPU or not
     <string> ckpt_dir: Model's location
    """

    # Preprocess images, returning the classes, features and labels
    class_names, x, y = preprocess_images(
        composed_dataset_path, 224, 224, num_classes, framework="torch", imagenet=True)

    # Split data
    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(
        x, y, folds)

    # Normalize
    X_train /= 255
    X_test /= 255

    # Instantiate model
    net = Net(
        models.vgg16(pretrained=True),
        num_classes=num_classes,
        cuda=cuda,
        mode="feature_composer",
        ckpt_dir=ckpt_dir
    )

    # Save labels for inference
    net.save_labels_for_inference(labels=class_names)

    # Train model
    net.fit(
        X_train,
        Y_train,
        X_test,
        Y_test,
        epochs,
        batch_size,
        resume=False
    )

    # Confusion matrix
    compute_confusion_matrix(y_true=Y_test, y_pred=net.infer(
        X_test), framework="torch", mode="feature_composer", num_classes = num_classes)

# Inference
def infer(ckpt_dir, ckpt_name, input_image):
    # Instantiate model
    net = Net(
        models.vgg16(pretrained=True),
        num_classes=num_classes,
        cuda=cuda,
        mode="feature_composer",
        ckpt_dir=ckpt_dir
    )

    # Load model
    net.load(os.path.join(ckpt_dir, ckpt_name))

    # Check if inputed file is an image.
    assert input_image.lower().endswith("png") or input_image.lower().endswith(
        "jpg") or input_image.lower().endswith("jpeg")

    # Preprocess
    img = preprocess_single_image(
        input_image, 224, 224, imagenet=True, framework="torch")

    # Return prediction
    return net.infer(img, use_labels=True)
