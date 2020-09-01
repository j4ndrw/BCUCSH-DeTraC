import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

from utils.preprocessing import preprocess_images
from utils.kfold import KFold_cross_validation_split
from utils.multiclass_confusion_matrix import multiclass_confusion_matrix

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
        cuda
    ):

    class_names, x, y = preprocess_images(initial_dataset_path, 224, 224, num_classes)

    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(x, y, folds)

    X_train /= 255
    X_test /= 255

    net = Net(models.vgg16(pretrained = True), num_classes = num_classes, cuda = cuda, mode = "feature_extractor")

    train_loss, train_acc, val_loss, val_acc = net.fit(
        X_train, 
        Y_train, 
        X_test, 
        Y_test, 
        epochs, 
        batch_size, 
        resume = False
    )

    for class_name in class_names:
        extracted_features = extract_features(initial_dataset_path, class_name, 224, 224, net)
        np.save(os.path.join(extracted_features_path, f"{class_name}.npy"), extract_features)

    compute_confusion_matrix(y_true = Y_test, y_pred = net.infer(X_test))

def extract_features(initial_dataset_path, class_name, width, height, net: Net):
    features = []

    for filename in tqdm(os.listdir(class_name)):
        gray_img = cv2.imread(os.path.join(initial_dataset_path, class_name, filename))
        color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(color_img, (width, height))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode = "torch")
        features.append(img)

    features = np.array(features)
    return net.infer_using_pretrained_layers_without_last(features)

def compute_confusion_matrix(y_true, y_pred):
    cmat = confusion_matrix(y_true.argmax(axis = 1), y_pred.argmax(axis = 1), normalize = "all")
    acc, sn, sp = multiclass_confusion_matrix(cmat, numClasses)

    output = f"ACCURACY = {acc}\nSENSITIVITY = {sn}\nSPECIFICITY = {sp}" 
    log_dir = "../../../models/torch/logs"
    with open(os.path.join(log_dir, "metrics_log"), 'w') as f:
        f.write(output)
    print(output)