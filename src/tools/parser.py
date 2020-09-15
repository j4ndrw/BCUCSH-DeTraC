import argparse
import sys

# Add framework description
parser = argparse.ArgumentParser(
    description="""Decompose. Transfer. Compose. 
    Before using this model, please put your image data in its appropriate folder inside the ./data/initial_data folder.
    Make sure each file is in its correspondent folder (e.g. COVID will have images of COVID-19 chest X-Ray images, etc...)
    Running the script for the first time will create some empty directories for you to put your data in it.""")

# Framework option
parser.add_argument(
    "-f", "--framework",
    metavar="[tf / tensorflow] / [torch / pytorch]",
    type=str,
    nargs=1,
    help="Chooses the used framework to run the DeTraC model",
    required=True
)

# Train option
parser.add_argument(
    "--train",
    action='store_const',
    const=True,
    default=False,
    help="Train the model"
)

# Epochs conditional option
parser.add_argument(
    "--epochs",
    required="--train" in sys.argv,
    metavar="N",
    type=int,
    nargs=1,
    help="Number of epochs in training loop"
)

# Batch size conditional option
parser.add_argument(
    "--batch_size",
    required="--train" in sys.argv,
    metavar="N",
    type=int,
    nargs=1,
    help="Batch size"
)

# Folds conditional option
parser.add_argument(
    "--folds",
    required="--train" in sys.argv,
    metavar="K",
    type=int,
    nargs=1,
    help="KFold Validation Split | Training Set = 100 - (K * 10) | Validation Set = (K * 10)"
)

# Number of classes conditional option
parser.add_argument(
    "--num_classes",
    required="--train" in sys.argv,
    metavar="N",
    type=int,
    nargs=1,
    help="Number of classes to train classifier on"
)

# Learning rate conditional option
parser.add_argument(
    "--lr",
    required="--train" in sys.argv,
    metavar="N",
    type=float,
    nargs=2,
    help="""Learning for feature extractor and feature composer.
    --lr [A B], where A is the learning rate for the feature extractor and B is the learning rate for the feature composer"""
)

# Learning rate conditional option
parser.add_argument(
    "--k",
    required="--train" in sys.argv,
    metavar="K",
    type=int,
    nargs=1,
    help="K-Means clustering"
)

# Inference option
parser.add_argument(
    "--infer",
    action='store_const',
    const=True,
    default=False,
    help="Use model for inference / prediction"
)

args = parser.parse_args()
