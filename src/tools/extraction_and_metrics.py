import os

import cv2

import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from .multiclass_confusion_matrix import multiclass_confusion_matrix

from tqdm import tqdm

# Feature extraction function
def extract_features(
    initial_dataset_path: str, 
    class_name: str, 
    width: int, 
    height: int,
    net,
    framework: str
) -> np.ndarray:
    """
    params:
        <string> initial_dataset_path = Path where initial data is located
        <string> class_name = Current class names
        <int, int> width, height = Image resolution
        <class<network.Net>> net = The network
    returns:
        <array> Features (4096 x 4096)
    """

    # Check if a framework is chosen
    assert framework == "tf" or framework == "torch"

    # Initialize features list
    features = []

    # Initialize progress bar for folder
    progress_bar = tqdm(os.listdir(os.path.join(initial_dataset_path, class_name)))
    progress_bar.set_description(f"Preparing {class_name} for feature extraction") 

    # Iterate through files in directory
    for filename in progress_bar:

        # Read grayscale image
        gray_img = cv2.imread(os.path.join(initial_dataset_path, class_name, filename))

        # Convert to RGB
        color_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)

        # Resize to the required size
        img = cv2.resize(color_img, (width, height))

        # Convert image to array
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Preprocess images for pretrained model
        if framework == "torch":
            img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode="torch")
        else:
            img = tf.keras.applications.imagenet_utils.preprocess_input(img, mode="tf")

        # Append image array to features list
        features.append(img)

    # Convert features list to vertical stack array
    features = np.vstack(features)

    # Return (Nx4096) prediction a.k.a extract features.
    return net.infer_using_pretrained_layers_without_last(features)

# Matlab equivalent of blockproc
def compose_classes(
    cmat: np.ndarray, 
    block_size: tuple
) -> np.ndarray:

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

# Metrics
def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    framework: str, 
    mode: str, 
    num_classes: int
):
    """
    params:
        <array> y_true: Ground truth labels
        <array> y_pred: Predicted labels
        <string> framework: Selected framework
        <string> mode: Feature extractor of feature composer mode
        <int> num_classes: Number of classes
    """

    # Check if a framework is chosen
    assert framework == "tf" or framework == "torch"

    # Check if a mode is selected
    assert mode == "feature_extractor" or mode == "feature_composer"
    
    # Create confusion matrix and normalize it
    cmat = confusion_matrix(
        y_true=y_true.argmax(axis=1), 
        y_pred=y_pred.argmax(axis=1), 
        normalize="all"
    )
    
    # If the feature composer was selected, divide the confusion matrix by NxN kernels
    if mode == "feature_composer":
        cmat = compose_classes(cmat, (2, 2))
        
    print(cmat)

    # Compute accuracy, sensitivity and specificity
    acc, sn, sp = multiclass_confusion_matrix(cmat, num_classes)
    output = f"ACCURACY = {acc}"
    print(output)

