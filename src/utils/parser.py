import argparse

parser = argparse.ArgumentParser(
    description = """Decompose. Transfer. Compose. 
    Before using this model, please put your image data in its appropriate folder inside the ./data/initial_data folder.
    Make sure each file is in it's correspondent folder (e.g. COVID will have images of COVID-19 chest X-Ray images, etc...)
    By default, there will be a dataset in the ./data/initial_data folder, but you can use your own if you wish.""")
parser.add_argument(
    "-f", "--framework", 
    metavar = "[tf / tensorflow] / [torch / pytorch]", 
    type = str, 
    nargs = 1, 
    help = "Chooses the used framework to run the DeTraC model",
    required = True
)

parser.add_argument(
    "--train", 
    action = 'store_const',
    const = True,
    default = False,
    help = "Train the model"
)

parser.add_argument(
    "--infer", 
    action = 'store_const',
    const = True,
    default = False,
    help = "Use model for inference / prediction"
)

args = parser.parse_args()