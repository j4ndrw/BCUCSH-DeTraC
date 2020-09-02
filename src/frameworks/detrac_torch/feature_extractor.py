import tensorflow as tf
import numpy as np

from utils.preprocessing import preprocess_images
from utils.kfold import KFold_cross_validation_split
from utils.extraction_and_metrics import extract_features, compute_confusion_matrix

from .network import Net

import torchvision.models as models
import torch

import os
import cv2


def train_feature_extractor(
    initial_dataset_path,
    extracted_features_path,
    epochs,
    batch_size,
    num_classes,
    folds,
    cuda,
    ckpt_dir
):

    class_names, x, y = preprocess_images(
        initial_dataset_path, 224, 224, num_classes, framework="torch")

    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(
        x, y, folds)

    X_train /= 255
    X_test /= 255

    net = Net(
        models.vgg16(pretrained=True),
        num_classes=num_classes,
        cuda=cuda,
        mode="feature_extractor",
        ckpt_dir=ckpt_dir
    )

    train_loss, train_acc, val_loss, val_acc = net.fit(
        X_train,
        Y_train,
        X_test,
        Y_test,
        epochs,
        batch_size,
        resume=False
    )

    for class_name in class_names:
        extracted_features = extract_features(
            initial_dataset_path, class_name, 224, 224, net, framework="torch")
        np.save(os.path.join(extracted_features_path,
                             f"{class_name}.npy"), extract_features)

    compute_confusion_matrix(y_true=Y_test, y_pred=net.infer(
        X_test), framework="torch", mode="feature_extractor")
