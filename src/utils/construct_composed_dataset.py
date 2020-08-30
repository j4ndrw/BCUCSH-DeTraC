import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

import cv2

def decompose(path_to_features, path_to_images, path_to_decomposed_images_1, path_to_decomposed_images_2):
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
    """
    features = np.load(path_to_features)
    
    idx = KMeans(n_clusters = 2, random_state=0).fit(features)
    idx = idx.predict(features)
    
    images = [filename for filename in os.listdir(path_to_images)] 
    
    for i in range(len(images)):
        filename = path_to_images + images[i]
        I = plt.imread(filename)

        filename_1 = path_to_decomposed_images_1 + images[i]
        filename_2 = path_to_decomposed_images_2 + images[i]
        if (idx[i] == 1):  
            plt.imsave(filename_1, I)
        else:
            plt.imsave(filename_2, I)

def execute_decomposition(initial_dataset_path, composed_dataset_path, features_path):
    assert os.path.exists(initial_dataset_path)
    assert os.path.exists(composed_dataset_path)
    assert os.path.exists(features_path)

    class_names = []
    for folder in os.listdir(initial_dataset_path):
        assert os.path.isdir(folder)
        class_names.append(folder)
    
    for class_name in class_names:
        # decomposition of normal class 
        try:
            os.mkdir(os.path.join("../data", composed_dataset_path, f"{class_name}_1/"))
        except:
            print("Directory {classname}_1 already exists")

        try:
            os.mkdir(os.path.join("../data", composed_dataset_path, f"{class_name}_2/"))
        except:
            print("Directory {classname}_2 already exists")

        decompose(
            path_to_features = os.path.join("../data", features_path, "{class_name}.npy"),
            path_to_images = os.path.join("../data", initial_dataset_path, class_name),
            path_to_decomposed_images_1 = os.path.join("../data", composed_dataset_path, f"{class_name}_1/")
            path_to_decomposed_images_2 = os.path.join("../data", composed_dataset_path, f"{class_name}_2/")
        )