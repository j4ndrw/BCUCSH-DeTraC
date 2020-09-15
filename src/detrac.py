from tools.parser import args
from tools import construct_composed_dataset

from frameworks import detrac_tf, detrac_torch
import numpy as np

import os

# Paths where data is stored
INITIAL_DATASET_PATH = "../data/initial_dataset"
EXTRACTED_FEATURES_PATH = "../data/extracted_features"
COMPOSED_DATASET_PATH = "../data/composed_dataset"

# Paths where models are stored
GENERAL_MODELS_PATH = "../models"
TF_MODEL_DIR = os.path.join(GENERAL_MODELS_PATH, "tf")
TORCH_CKPT_DIR = os.path.join(GENERAL_MODELS_PATH, "torch")

TF_MODEL_DETAILS_DIR = os.path.join(GENERAL_MODELS_PATH, TF_MODEL_DIR, "details")

# TODO: Document and proofread
# TODO: Fix / implement save and resume mechanic OR get rid of it (not ideal)
# TODO: Implement different log mechanic for number of classes and labels (TF)
# TODO: Test inference on images (TF).
# TODO: Write appropriate readme
# TODO: Create setup.py and build.py

# Training option
def training(args):
    """
    DeTraC training function:
    1) Trains feature extractor
    2) Composes a new dataset using the features extracted by a pretrained model, with a custom classification layer
    3) Trains a feature composer, that takes those composed images and classifies them appropriately

    params:
        <list> args: Arguments entered by user
    """

    # Get training options from argument parser
    num_epochs = args.epochs[0]
    batch_size = args.batch_size[0]
    feature_extractor_num_classes = args.num_classes[0]
    feature_composer_num_classes = 2 * feature_extractor_num_classes
    folds = args.folds[0]
    k = args.k[0]
    feature_extractor_lr = args.lr[0]
    feature_composer_lr = args.lr[1]

    # If user chose "Tensorflow" for the framework option
    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # Train the feature extractor
        detrac_tf.feature_extractor.train_feature_extractor(
            initial_dataset_path=INITIAL_DATASET_PATH,
            extracted_features_path=EXTRACTED_FEATURES_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_extractor_num_classes,
            folds=folds,
            lr=feature_extractor_lr,
            model_dir=TF_MODEL_DIR
        )

        # Construct the dataset composed using the extracted features
        construct_composed_dataset.execute_decomposition(
            initial_dataset_path=INITIAL_DATASET_PATH,
            composed_dataset_path=COMPOSED_DATASET_PATH,
            features_path=EXTRACTED_FEATURES_PATH,
            k=k
        )

        # Train feature composer on composed dataset
        detrac_tf.feature_composer.train_feature_composer(
            composed_dataset_path=COMPOSED_DATASET_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_composer_num_classes,
            folds=folds,
            lr=feature_composer_lr,
            model_dir=TF_MODEL_DIR
        )

    # If user chose "Pytorch" for the framework option
    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        # Prompt them to choose a method of computation
        # (In TF, if tensorflow-gpu is installed, this is inferred, though, in Pytorch it is done manually)
        use_cuda = input("Use CUDA for GPU computation? [Y / N]: ")
        if use_cuda.lower() == "y" or use_cuda.lower() == "yes":
            use_cuda = True
        elif use_cuda.lower() == "n" or use_cuda.lower() == "no":
            use_cuda = False

        # Train the feature extractor
        detrac_torch.feature_extractor.train_feature_extractor(
            initial_dataset_path=INITIAL_DATASET_PATH,
            extracted_features_path=EXTRACTED_FEATURES_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_extractor_num_classes,
            folds=folds,
            lr=feature_extractor_lr,
            cuda=use_cuda,
            ckpt_dir=TORCH_CKPT_DIR
        )

        # Construct the dataset composed using the extracted features
        construct_composed_dataset.execute_decomposition(
            initial_dataset_path=INITIAL_DATASET_PATH,
            composed_dataset_path=COMPOSED_DATASET_PATH,
            features_path=EXTRACTED_FEATURES_PATH,
            k=k
        )

        # Train feature composer on composed dataset
        detrac_torch.feature_composer.train_feature_composer(
            composed_dataset_path=COMPOSED_DATASET_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_composer_num_classes,
            folds=folds,
            lr=feature_composer_lr,
            cuda=use_cuda,
            ckpt_dir=TORCH_CKPT_DIR
        )

# Inference option
def inference(args):
    """
    DeTraC inference function:
    Prompt user with a path where the image file is located. It will then output the predicted class, along with it's level of confidence.

    params:
        <list> args: Arguments entered by user
    """
    
    # TODO: Refactor load mechanic

    # Prompt the user with the option to choose a file
    path_to_file = input(
        "Please enter the path of the file you wish to run the model upon (e.g.: /path/to/image.png): ")

    print(path_to_file)
    # Check if file exists
    assert os.path.exists(path_to_file)

    # Check if file is an image (no GIFs)
    assert path_to_file.lower().endswith(".png") or path_to_file.lower().endswith(".jpg") or path_to_file.lower().endswith(".jpeg")

    # If user chose "Tensorflow" for the framework option
    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # Create a cache containing all trained models
        model_list = []
        print("Here is a list of your models: ")
        idx = 1
        for model in os.listdir(TF_MODEL_DIR):
            if "feature_composer" in model:
                print(f"{idx}) {model}")
                idx += 1
                model_list.append(model)

        # Prompt user to choose a model
        model_choice = -1
        while model_choice > len(model_list) or model_choice < 1:
            model_choice = int(input(f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

        # Predict
        prediction = detrac_tf.feature_composer.infer(
            model_details_dir=TF_MODEL_DETAILS_DIR,
            model_dir=TF_MODEL_DIR,
            model_name=model_list[model_choice - 1], 
            input_image=path_to_file
        )
        print(f"Prediction: {list(prediction.keys())[0].split('_')[0]}")
        print(f"Confidence: \n{prediction}")

    # If user chose "Pytorch" for the framework option
    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        # Create a cache containing all trained models
        model_list = []
        print("Here is a list of your models: ")
        idx = 1
        for model in os.listdir(TORCH_CKPT_DIR):
            if "feature_composer" in model:
                print(f"{idx}) {model}")
                idx += 1
                model_list.append(model)

        assert len(model_list) != 0
                
        # Prompt user to choose a model
        model_choice = -1
        while model_choice > len(model_list) or model_choice < 1:
            model_choice = int(input(f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

        # Predict
        prediction = detrac_torch.feature_composer.infer(
            ckpt_dir=TORCH_CKPT_DIR, 
            ckpt_name=model_list[model_choice - 1], 
            input_image=path_to_file)

        print(f"Prediction: {list(prediction.keys())[np.array(list(prediction.values())).argmax()].split('_')[0]}")
        print(f"Confidence: \n{prediction}")

# Function used to initialize repo with the necessary folders.
def init_folders(path: str) -> bool:
    """
    Used to initialize folders if there aren't already there

    params:
        <string> path

    returns:
        <bool>
    """

    if not os.path.exists(path):
        print(f"{path} doesn't exist. Initializing...")
        os.mkdir(path)
        return True
    return False

def main():
    fresh_directories = [
        init_folders(INITIAL_DATASET_PATH),
        init_folders(EXTRACTED_FEATURES_PATH),
        init_folders(COMPOSED_DATASET_PATH),
        init_folders(GENERAL_MODELS_PATH),
        init_folders(TF_MODEL_DETAILS_DIR)
    ]

    if all(fresh_directories) == True:
        print(f"The directories have just been created. Make sure to populate the {INITIAL_DATASET_PATH} with your data.")
        exit(0)
    else:
        if len(os.listdir(INITIAL_DATASET_PATH)) == 0:
            print(f"Your main data directory ({INITIAL_DATASET_PATH}) is empty. Make sure to populate it before running the script.")
            exit(0)

    # Choice of framework
    option = args.framework[0].lower()
    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # Use TensorFlow
        print("\n[Tensorflow Backend]\n")
        init_folders(TF_MODEL_DIR)
    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        # Use PyTorch
        print("\n[PyTorch Backend]\n")
        init_folders(TORCH_CKPT_DIR)

    # Mode selection.
    # If no mode is selected, exit
    if args.train == False and args.infer == False:
        # No option = No reason to use the model
        print("No option selected.")
        exit(0)

    # If one or both modes were selected
    else:
        # If both the training mode and the inference mode are selected
        if args.train == True and args.infer == True:
            print("\nPreparing the model for training and inference\n")

            # Train
            training(args)

            # Infer
            inference(args)
        else:
            # If only the training mode was selected
            if args.train == True and args.infer == False:
                print("\nPreparing the model for training\n")

                # Train
                training(args)
            # Otherwise
            elif args.train == False and args.infer == True:
                print("\nPreparing the model for inference\n")

                # Infer
                inference(args)


if __name__ == "__main__":
    main()
