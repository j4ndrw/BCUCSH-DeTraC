from utils.parser import args
from utils import construct_composed_dataset

from frameworks import detrac_tf, detrac_torch

import os

INITIAL_DATASET_PATH = "../data/initial_dataset"
EXTRACTED_FEATURES_PATH = "../data/extracted_features"
COMPOSED_DATASET_PATH = "../data/composed_dataset"

TF_MODEL_DIR = "../models/tf"
TORCH_CKPT_DIR = "../models/torch"

# TODO: Document and proofread
# TODO: Fix / implement save and resume mechanic OR get rid of it (not ideal)
# TODO: Do more in-depth testing
# TODO: Test on new data. Add whatever you added to tf backend onto torch backend.
# TODO: Fix more stuff
# TODO: Add args as alternative to prompts
# TODO: Test inference on images.

def training(args):
    num_epochs = int(input("Number of epochs: "))
    batch_size = int(input("Batch size: "))
    feature_extractor_num_classes = int(
        input("How many classes are there to predict?: "))
    feature_composer_num_classes = 2 * feature_extractor_num_classes
    k = int(input(
        "How do you wish to split the data? [KFold Validation Split]\nTraining Set = 100% - (K * 10)%\nValidation Set = (K * 10)%\nK = "))

    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        detrac_tf.feature_extractor.train_feature_extractor(
            initial_dataset_path=INITIAL_DATASET_PATH,
            extracted_features_path=EXTRACTED_FEATURES_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_extractor_num_classes,
            folds=k,
            model_dir=TF_MODEL_DIR
        )

        construct_composed_dataset.execute_decomposition(
            initial_dataset_path=INITIAL_DATASET_PATH,
            composed_dataset_path=COMPOSED_DATASET_PATH,
            features_path=EXTRACTED_FEATURES_PATH
        )

        detrac_tf.feature_composer.train_feature_composer(
            composed_dataset_path=COMPOSED_DATASET_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_composer_num_classes,
            folds=k,
            model_dir=TF_MODEL_DIR
        )

    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        use_cuda = input("Use CUDA for GPU computation? [Y / N]: ")
        if use_cuda.lower() == "y" or use_cuda.lower() == "yes":
            use_cuda = True
        elif use_cuda.lower() == "n" or use_cuda.lower() == "no":
            use_cuda = False

        detrac_torch.feature_extractor.train_feature_extractor(
            initial_dataset_path=INITIAL_DATASET_PATH,
            extracted_features_path=EXTRACTED_FEATURES_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_extractor_num_classes,
            folds=k,
            cuda=use_cuda,
            ckpt_dir=TORCH_CKPT_DIR
        )

        construct_composed_dataset.execute_decomposition(
            initial_dataset_path=INITIAL_DATASET_PATH,
            composed_dataset_path=COMPOSED_DATASET_PATH,
            features_path=EXTRACTED_FEATURES_PATH
        )

        detrac_torch.feature_composer.train_feature_composer(
            composed_dataset_path=COMPOSED_DATASET_PATH,
            epochs=num_epochs,
            batch_size=batch_size,
            num_classes=feature_composer_num_classes,
            cuda=use_cuda,
            ckpt_dir=TORCH_CKPT_DIR
        )


def inference(args):
    path_to_file = input(
        "Please enter the path of the file you wish to run the model upon: ")
    assert os.path.exists(path_to_file)
    assert path_to_file.lower().endswith(".png") or path_to_file.lower().endswith(
        ".jpg") or path_to_file.lower().endswith(".jpeg")

    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        model_list = []
        print("Here is a list of your models: ")
        for i, model in enumerate(os.listdir(TF_MODEL_DIR)):
            if "feature_composer" in model:
                print(f"{i + 1}) {model}")
                model_list.append(model)

        model_choice = -1
        while model_choice > len(model_list) or model_choice < 1:
            model_choice = int(input(
                f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

        prediction = detrac_tf.feature_composer.infer(
            TF_MODEL_DIR, model_list[model_choice - 1], path_to_file)

        print(f"Prediction: {list(prediction.keys())[0]}")
        print(f"Confidence: {list(prediction.values())}")

    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        model_list = []
        print("Here is a list of your models: ")
        for i, model in enumerate(os.listdir(TORCH_CKPT_DIR)):
            if "feature_composer" in model:
                print(f"{i + 1}) {model}")
                model_list.append(model)

        model_choice = -1
        while model_choice > len(model_list) or model_choice < 1:
            model_choice = int(input(
                f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

        prediction = detrac_torch.feature_composer.infer(
            TORCH_CKPT_DIR, model_list[model_choice - 1], path_to_file)

        print(f"Prediction: {list(prediction.keys())[0]}")
        print(f"Confidence: {list(prediction.values())}")


def main():
    option = args.framework[0].lower()
    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # Use TensorFlow
        print("\n[Tensorflow Backend]\n")

    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        # Use PyTorch
        print("\n[PyTorch Backend]\n")

    if option == 'tf' or option == 'tensorflow':
        if (args.train == None and args.infer == None) or (args.train == False and args.infer == False):
            # No option = No reason to use the model
            print("No option selected.")
            exit(0)

        else:
            if args.train == True and args.infer == True:
                # Training + Inference mode
                print("\nPreparing the model for training and inference\n")
                training(args)
                inference(args)
            else:
                if args.train == True and args.infer == False:
                    # Training mode
                    print("\nPreparing the model for training\n")
                    training(args)
                elif args.train == False and args.infer == True:
                    # Inference mode
                    print("\nPreparing the model for inference\n")
                    inference(args)
    if option == 'torch' or option == 'pytorch':
        from frameworks import detrac_torch
        if (args.train == None and args.infer == None) or (args.train == False and args.infer == False):
            # No option = No reason to use the model
            print("No option selected.")
            exit(0)

        else:
            if args.train == True and args.infer == True:
                # Training + Inference mode
                print("\nPreparing the model for training and inference\n")
                training(args)
                inference(args)
            else:
                if args.train == True and args.infer == False:
                    # Training mode
                    print("\nPreparing the model for training")
                    training(args)
                elif args.train == False and args.infer == True:
                    # Inference mode
                    print("\nPreparing the model for inference\n")
                    inference(args)


if __name__ == "__main__":
    main()
