import numpy as np
from sklearn.cluster import KMeans
import os
import shutil
import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

def decompose(
    path_to_features: str, 
    path_to_images: str, 
    path_to_decomposed_images_1: str, 
    path_to_decomposed_images_2: str, 
    class_name: str,
    k: int
):
    """
    Decomposition of extracted features using KMeans clustering.

    Algorithm:
        - Loads the features
        - Creates 2 clusters based on the data points
        - Each cluster corresponds in its own data subfolder

    params:
        <string> path_to_features
        <string> path_to_images
        <string> path_to_decomposed_images_1
        <string> path_to_decomposed_images_2
        <int> k: Number of clusters
    """

    # Load features
    features = np.load(path_to_features)

    # Cluster index
    idx = KMeans(n_clusters=k, random_state=0).fit(features)
    idx = idx.predict(features)

    # Images list
    images = [filename for filename in os.listdir(path_to_images)]

    # Iterate through images
    progress_bar = tqdm(range(len(images)))
    progress_bar.set_description(f"Composing {class_name} images")
    for i in progress_bar:
        filename = os.path.join(path_to_images, images[i])
        
        # Read image
        I = plt.imread(filename)

        filename_1 = os.path.join(path_to_decomposed_images_1, images[i])
        filename_2 = os.path.join(path_to_decomposed_images_2, images[i])
        
        # If image belongs to a cluster, write the image to a certain folder, otherwise, write it to the other folder.
        if (idx[i] == 1):
            plt.imsave(filename_1, I)
        else:
            plt.imsave(filename_2, I)

def execute_decomposition(
    initial_dataset_path: str, 
    composed_dataset_path: str, 
    features_path: str,
    k:int
):
    """
    Decomposes features in a separate dataset

    params:
        <string> initial_dataset_path
        <string> composed_dataset_path
        <string> features_path
        <int> k: Number of clusters
    """

    # Check if folders exist
    assert os.path.exists(initial_dataset_path)
    assert os.path.exists(composed_dataset_path)
    assert os.path.exists(features_path)

    # Initialize list of classes
    class_names = []

    # Check if folders exist and add them to the list of classes
    for folder in os.listdir(initial_dataset_path):
        assert os.path.isdir(os.path.join(initial_dataset_path, folder))
        class_names.append(folder)

    # For every class
    for class_name in class_names:
        try:
            # Create folder for 1st cluster.
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_1/"))
        except:
            # If it cannot create a folder, it will overwrite it.
            print(f"Directory {class_name}_1 already exists. Overwriting.")
            shutil.rmtree(os.path.join(composed_dataset_path, f"{class_name}_1/"))
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_1/"))
        try:
            # Create folder for 2nd cluster
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_2/"))
        except:
            # If it cannot create a folder, it will overwrite it.
            print(f"Directory {class_name}_2 already exists. Overwriting.")
            shutil.rmtree(os.path.join(composed_dataset_path, f"{class_name}_2/"))
            os.mkdir(os.path.join(composed_dataset_path, f"{class_name}_2/"))

        # Decompose said class' data
        decompose(
            path_to_features=os.path.join(features_path, f"{class_name}.npy"),
            path_to_images=os.path.join(initial_dataset_path, class_name),
            path_to_decomposed_images_1=os.path.join(composed_dataset_path, f"{class_name}_1/"),
            path_to_decomposed_images_2=os.path.join(composed_dataset_path, f"{class_name}_2/"),
            class_name=class_name,
            k=k
        )
