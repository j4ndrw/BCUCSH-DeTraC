import tensorflow as tf

from tensorflow.keras.applications import VGG16

import numpy as np

from tools.preprocessing import preprocess_images
from tools.kfold import KFold_cross_validation_split
from tools.extraction_and_metrics import extract_features, compute_confusion_matrix

from .network import Net

import os
import cv2

# Feature composer training
def train_feature_composer(
    composed_dataset_path: str,
    epochs: int,
    batch_size: int,
    num_classes: int,
    folds: int,
    model_dir: str
):
    """
    Feature extractor training.

    params:
     <string> composed_dataset_path
     <int> epochs
     <int> batch_size
     <int> num_classes
     <int> folds: Number of folds for KFold cross validation 
     <string> model_dir: Model's location
    """

    # Preprocess images, returning the classes, features and labels
    class_names, x, y = preprocess_images(
        dataset_path=composed_dataset_path, 
        width=224, 
        height=224, 
        num_classes=num_classes, 
        framework="tf", 
        imagenet=True
    )

    # Split data
    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(
        features=x, 
        labels=y, 
        n_splits=folds
    )

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
        mode="feature_composer",
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

    # Confusion matrix
    compute_confusion_matrix(
        y_true=Y_test, 
        y_pred=net.infer(X_test), 
        framework="tf", 
        mode="feature_composer", 
        num_classes=num_classes // 2
    )

# Inference
def infer(
    model_dir: str,
    model_name: str,
    input_image: str
) -> dict:

    # Instantiate model
    net = Net(
        pretrained_model=VGG16(
            input_shape=(224, 224, 3),
            include_top=True
        ),
        num_classes=num_classes,
        mode="feature_composer",
        class_names=class_names,
        model_dir=model_dir
    )

    # Load model
    tf.keras.models.load_model(
        filepath=os.path.join(model_dir, model_name), 
        compile=False
    )

    # Check if inputed file is an image
    assert input_image.lower().endswith("png") or input_image.lower().endswith("jpg") or input_image.lower().endswith("jpeg")

    # Preprocess
    img = preprocess_single_image(
        img = input_image, 
        width=224, 
        height=224, 
        imagenet=True, 
        framework="tf"
    )

    # Prediction
    return net.infer(img, use_labels=True)
