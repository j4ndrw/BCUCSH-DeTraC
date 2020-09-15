import tensorflow as tf

from tensorflow.keras.applications import VGG16

import numpy as np

from tools.preprocessing import preprocess_images, preprocess_single_image
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
    lr: float,
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
     <float> lr: Learning rate
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
        lr=lr,
        mode="feature_composer",
        labels=class_names,
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
    model_details_dir: str,
    model_dir: str,
    model_name: str,
    input_image: str
) -> dict:
    """
    Main inference method:

    params:
        <string> model_details_dir: Saved model's details directory 
                                    (contains label names and number of classes - to be loaded for inference)
        <string> model_dir: Saved model's directory
        <string> model_name: Saved model's name
        <string> input_image: Image path

    returns:
        <dict> Dictionary containing the predictions with their levels of confidence.
                E.g.: {
                    COVID19_1:0.10
                    COVID19_2:0.15
                    ...
                }
    """

    with open(f"{os.path.join(model_details_dir, f'{model_name}.detrac')}", "r") as f:
        details = f.read()
    
    num_classes = int(details.split("-|-")[-1])
    labels = details.split("-|-")[:-1]
    
    # Instantiate model
    net = Net(
        pretrained_model=VGG16(
            input_shape=(224, 224, 3),
            include_top=True
        ),
        num_classes=num_classes,
        mode="feature_composer",
        model_dir=model_dir,
        labels=labels
    )

    # Load model
    tf.keras.models.load_model(
        filepath=os.path.join(model_dir, model_name)
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
