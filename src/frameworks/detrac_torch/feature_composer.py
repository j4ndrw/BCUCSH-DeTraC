import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

from utils.preprocessing import preprocess_images, preprocess_single_image_torch
from utils.kfold import KFold_cross_validation_split
from utils.multiclass_confusion_matrix import multiclass_confusion_matrix

from .network import Net

import torchvision.models as models
import torch

import os
import cv2

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
    composed = reshaped_matrix.sum(axis = (1, 3))
    return composed

def compute_confusion_matrix(y_true, y_pred):
    cmat = confusion_matrix(y_true.argmax(axis = 1), y_pred.argmax(axis = 1), normalize = "all")
    cmat = compose_classes(cmat, (2, 2))
    acc, sn, sp = multiclass_confusion_matrix(cmat, numClasses)

    output = f"ACCURACY = {acc}\nSENSITIVITY = {sn}\nSPECIFICITY = {sp}" 
    log_dir = "../../../models/torch/logs"
    with open(os.path.join(log_dir, "metrics_log"), 'w') as f:
        f.write(output)
    print(output)

def train_feature_composer(
        composed_dataset_path,
        epochs, 
        batch_size, 
        num_classes, 
        cuda
    ):

    class_names, x, y = preprocess_images(composed_dataset_path, 224, 224, num_classes, True)

    X_train, X_test, Y_train, Y_test = KFold_cross_validation_split(x, y)

    X_train /= 255
    X_test /= 255

    net = Net(models.vgg16(pretrained = True), num_classes = num_classes, cuda = cuda,mode = "feature_composer")

    net.save_labels_for_inference(labels = class_names)

    train_loss, train_acc, val_loss, val_acc = net.fit(
        X_train, 
        Y_train, 
        X_test, 
        Y_test, 
        epochs, 
        batch_size, 
        resume = False
    )

    compute_confusion_matrix(y_true = Y_test, y_pred = net.infer(X_test))

def infer(checkpoint_path, input_image):
    net = Net(models.vgg16(pretrained = True), num_classes = num_classes, cuda = cuda, mode = "feature_composer")
    net.load(checkpoint_path)
    assert input_image.lower().endswith("png") or input_image.lower().endswith("jpg") or input_image.lower().endswith("jpeg")
    img = preprocess_single_image_torch(input_image, 224, 224, imagenet = True)
    return net.infer(img)