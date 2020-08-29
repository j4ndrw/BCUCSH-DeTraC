from parser import args

import os

def training(args):
    num_epochs = int(input("Number of epochs: "))
    batch_size = int(input("Batch size: "))
    feature_extractor_num_classes = int(input("How many classes are there to predict?: "))
    feature_composer_num_classes = 2 * feature_extractor_num_classes
    kfold = int(input("How do you wish to split the data? [KFold Cross Validation = K% Training Set and (100% - K%) Validation Set]: "))

    if args.framework[0].lower() == "tf" or args.framework[0].lower() == "tensorflow":
        from frameworks import detrac_tf
        
    elif args.framework[0].lower() == "torch" or args.framework[0].lower() == "pytorch":
        use_cuda = input("Use CUDA for GPU computation? [Y / N]: ")
        from frameworks import detrac_torch
        if use_cuda.lower() == "y" or use_cuda.lower() == "yes":
            use_cuda = True
        elif use_cuda.lower() == "n" or use_cuda.lower() == "no":
            use_cuda = False

def inference(args):
    path_to_file = input("Please enter the path of the file you wish to run the model upon: ")
    assert os.path.exists(path_to_file)
    assert path_to_file.lower().endswith(".png") or path_to_file.lower().endswith(".jpg") or path_to_file.lower().endswith(".jpeg")

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
                training_mode()
                inference_mode()
            else:
                if args.train == True and args.infer == False:
                    # Training mode
                    print("\nPreparing the model for training\n")
                    training_mode()
                elif args.train == False and args.infer == True:
                    # Inference mode
                    print("\nPreparing the model for inference\n")
                    inference_mode()
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
                training_mode()
                inference_mode()
            else:
                if args.train == True and args.infer == False:
                    # Training mode
                    print("\nPreparing the model for training")
                    training_mode()
                elif args.train == False and args.infer == True:
                    # Inference mode
                    print("\nPreparing the model for inference\n")
                    inference_mode()

if __name__ == "__main__":
    main()