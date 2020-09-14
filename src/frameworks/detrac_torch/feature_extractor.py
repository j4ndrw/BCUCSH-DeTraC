import tensorflow as tf
import numpy as np

from tools.preprocessing import preprocess_images
from tools.kfold import KFold_cross_validation_split
from tools.extraction_and_metrics import extract_features, compute_confusion_matrix

from .network import Net

import torchvision.models as models
import torch

import os
import cv2

# Feature extractor training
def train_feature_extractor(
    initial_dataset_path: str,
    extracted_features_path: str,
    epochs: int,
    batch_size: int,
    num_classes: int,
    folds: int,
    lr:float,
    cuda: bool,
    ckpt_dir: str
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
     <float> lr: Learning rate
     <bool> cuda: Whether to use GPU or not
     <string> ckpt_dir: Model's location
    """

    # Preprocess images, returning the classes, features and labels
    class_names, x, y = preprocess_images(
        dataset_path=initial_dataset_path, 
        width=224, 
        height=224, 
        num_classes=num_classes, 
        framework="torch"
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
        models.vgg16(pretrained=True),
        num_classes=num_classes,
        lr=lr,
        cuda=cuda,
        mode="feature_extractor",
        ckpt_dir=ckpt_dir,
        labels=class_names
    )

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
    compute_confusion_matrix(
        y_true=Y_test, 
        y_pred=net.infer(X_test), 
        framework="torch", 
        mode="feature_extractor", 
        num_classes = num_classes
    )
    
    # Extract features
    for class_name in class_names:
        extracted_features = extract_features(
            initial_dataset_path=initial_dataset_path, 
            class_name=class_name, 
            width=224, 
            height=224, 
            net=net, 
            framework="torch"
        )
        np.save(
            file=os.path.join(extracted_features_path, f"{class_name}.npy"), 
            arr=extracted_features
        )