from utils.parser import args
from utils import construct_composed_dataset

import os

def training(args):
    num_epochs = int(input("Number of epochs: "))
    batch_size = int(input("Batch size: "))
    feature_extractor_num_classes = int(input("How many classes are there to predict?: "))
    feature_composer_num_classes = 2 * feature_extractor_num_classes
    k = int(input("How do you wish to split the data? [KFold Cross Validation = K% Training Set and (100% - K%) Validation Set]: "))

    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # TODO: Implement this.
        pass

    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        use_cuda = input("Use CUDA for GPU computation? [Y / N]: ")
        from frameworks import detrac_torch
        if use_cuda.lower() == "y" or use_cuda.lower() == "yes":
            use_cuda = True
        elif use_cuda.lower() == "n" or use_cuda.lower() == "no":
            use_cuda = False

        detrac_torch.feature_extractor.train_feature_extractor(
            "data/initial_dataset",
            epochs = num_epochs,
            batch_size = batch_size,
            num_classes = feature_extractor_num_classes,
            folds = k,
            cuda = use_cuda
        )

        construct_composed_dataset.execute_decomposition(
            "../data/initial_dataset",
            "../data/composed_dataset",
            "../data/extracted_features"
        )

        detrac_torch.feature_composer.train_feature_composer(
            "data/composed_dataset",
            epochs = num_epochs,
            batch_size = batch_size,
            num_classes = feature_composer_num_classes,
            cuda = use_cuda
        )

def inference(args):
    path_to_file = input("Please enter the path of the file you wish to run the model upon: ")
    assert os.path.exists(path_to_file)
    assert path_to_file.lower().endswith(".png") or path_to_file.lower().endswith(".jpg") or path_to_file.lower().endswith(".jpeg")

    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # TODO: Implement this.
        pass

    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        from frameworks import detrac_torch
        model_list = []
        path_to_models = "../models/torch"
        print("Here is a list of your models: ")
        for i, model in enumerate(os.listdir(path_to_models)):
            if model.endswith("pth"):
                print(f"{i}) {model}")
        
        model_choice = -1
        while model_choice > len(model_list) or model_choice < 1:
            model_choice = int(input(f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

        path_to_model = os.path.join(path_to_models, model_list[model_choice - 1])

        prediction = detrac_torch.feature_composer.infer(path_to_model, path_to_file)
        # TODO: Save labels in model checkpoint file for inference.

def main():
    option = args.framework[0].lower()
    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        # Use TensorFlow
        print("\n[Tensorflow Backend]\n")

    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        # Use PyTorch
        print("\n[PyTorch Backend]\n")

    print(option)
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
