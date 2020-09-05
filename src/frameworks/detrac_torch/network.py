import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np

import os
import time
from datetime import datetime

from tqdm import tqdm

# Function used to choose computation device.
def set_device(var, use_cuda):
    """
    Gets tensor or nn variable ready for computation on a device.

    params:
        <tensor or nn> var: Tensor or model
        <bool> use_cuda: Whether to use GPU for computation or not

    returns:
        <tensor or nn> var: The same tensor or model, prepared for computation on the selected device
    """
    if use_cuda == True:
        if torch.cuda.is_available():
            var = var.cuda()
        else:
            var = var.cpu()
    else:
        var = var.cpu()

    return var

# Custom dataset used for augmentation
class augmented_data(Dataset):
    """
    Augmented dataset

    This class inherits from PyTorch's Dataset class, which allows for overloading the initializer and getter to contain a transform operation.
    """

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item

# The network
class Net(object):
    """
    The DeTraC model.
    """

    def __init__(
        self,
        pretrained_model,
        num_classes: int,
        cuda: bool,
        mode: str,
        ckpt_dir: str
    ):
        """
        params:
            <inherits nn.Module> pretrained_model: VGG, AlexNet or whatever other ImageNet pretrained model is chosen
            <int> num_classes
            <bool> cuda: Choice of computation device (Whether to use GPU or not)
            <string> mode: The DeTraC model contains 2 modes which are used depending on the case:
                                - feature_extractor: used in the first phase of computation, where the pretrained model is used to extract the main features from the dataset
                                - feature_composer: used in the last phase of computation, where the model is now training on the composed images, using the extracted features and clustering them.
        """

        self.mode = mode
        self.model = pretrained_model
        self.num_classes = num_classes
        self.cuda = cuda
        self.ckpt_dir = ckpt_dir

        # Check whether the mode is correct
        assert self.mode == "feature_extractor" or self.mode == "feature_composer"

        # Check if checkpoint directory exists
        assert os.path.exists(ckpt_dir)

        # Prepare model for computation on the selected device
        self.model = set_device(self.model, self.cuda)


        # Extract the input size based on the second to last layer
        try:
            self.input_size = self.model.classifier[-1].in_features
        except:
            self.input_size = self.model.fc.in_features

        # Introduce a new layer of computation
        self.classification_layer = nn.Linear(
            in_features=self.input_size, out_features=self.num_classes)
        self.softmax_activation = nn.Softmax(dim=1)

        # Set the weights and biases accordingly
        with torch.no_grad():
            self.classification_layer.weight = torch.nn.Parameter(set_device(torch.randn((self.num_classes, self.input_size)) * 1e-5, self.cuda))
            self.classification_layer.bias = torch.nn.Parameter(set_device(torch.randn(self.num_classes) * 1e-5 + 1, self.cuda))

        # Prepare the layer for computation on the selected device
        self.classification_layer = set_device(nn.Sequential(
            self.classification_layer, self.softmax_activation), self.cuda)

        # Replace the pretrained classification layer with the custom classification layer
        try:
            self.model.classifier[-1] = self.classification_layer
        except:
            self.model.fc = self.classification_layer

        # Set the save path, freeze or unfreeze the gradients based on the mode and define appropriate optimizers and schedulers.
        # Feature extractor => Freeze all gradients except the custom classification layer
        # Feature composer => Unfreeze / Activate all gradients
        now = datetime.now()
        now = f'{str(now).split(" ")[0]}_{str(now).split(" ")[1]}'.split(
            ".")[0].replace(':', "-")
        if self.mode == "feature_extractor":
            self.save_name = f"DeTraC_feature_extractor_{now}.pth"

            for param in self.model.parameters():
                param.requires_grad = False

            try:
                for param in self.model.classifier[-1].parameters():
                    param.requires_grad = True
            except:
                for param in self.model.fc.parameters():
                    param.requires_grad = True

            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=1e-4,
                momentum=0.9,
                nesterov=False,
                weight_decay=1e-3
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=0.9,
                patience=3
            )

        else:
            self.save_name = f"DeTraC_feature_composer_{now}.pth"

            for param in self.model.parameters():
                param.requires_grad = True

            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=1e-4,
                momentum=0.95,
                nesterov=False,
                weight_decay=1e-4
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=0.95,
                patience=5
            )

        self.ckpt_path = os.path.join(self.ckpt_dir, self.save_name)

        # Define the loss.
        # Categorical crossentropy is a negative log likelihood loss, where the logit is the log of the model's output, and the label is the argmax from the list of labels
        self.criterion = nn.NLLLoss()

    def save(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """
        Save the model's gradients, as well as the optimizer's latent gradients.
        Also save some additional data, such as epoch, loss and accuracy.

        params:
            <int> epoch
            <float> train_loss
            <float> train_acc
            <float> val_loss
            <float> val_acc
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }, self.ckpt_path)

    def load(self, *args):
        """
        Load the model on GPU or CPU.

        params:
            <unpacked_list> args

        returns:
            <list> loaded_args: The arguments used in the model's checkpoint, used to load the model. 
        """
        
        # Prompt the user
        prompt = input("Load on GPU or CPU [GPU / CPU]\n")
        while prompt != "GPU" and prompt != "CPU":
            prompt = input("Load on GPU or CPU? [GPU / CPUs]\n")

        # Check if model exists
        assert os.path.exists(self.ckpt_path)

        print("Loading checkpoint")
        # Load model on selected computation device
        if prompt == "CPU":
            checkpoint = torch.load(
                self.ckpt_path, map_location=lambda storage, loc: storage)
            self.cuda = False
        else:
            checkpoint = torch.load(self.ckpt_path)
            self.cuda = True

        # List of arguments to load
        loaded_args = []
        loaded_model = False
        for arg in args:
            if arg == "model_state_dict":
                if loaded_model == False:
                    print("Loading model state")
                    self.model.load_state_dict(checkpoint[arg])
                    loaded_model = True
            else:
                try:
                    print(f"Loading {arg}")
                    loaded_args.append(checkpoint[arg])
                except:
                    print(f"{arg} does not exist in model checkpoint.")

        if len(loaded_args) != 0:
            return loaded_args

    def save_labels_for_inference(self, labels):
        """
        Saves labels in the model's checkpoint.
        Purpose: for the user to not have the possibility of accidentally deleting a log file where labels may be or for the user to have details about the label.
        E.g.: It's much easier to see "PNEUMONIA", rather than [0.1, 0.2, 0.7]

        params:
            <list> labels: One-hot encoded labels
        """

        torch.save({
            "labels": labels
        }, self.ckpt_path)

    def load_labels_for_inference(self):
        """
        Load labels from the model's checkpoint 
        """

        # Check if checkpoint exists
        assert os.path.exists(self.ckpt_path)
        checkpoint = torch.load(
            self.ckpt_path, map_location=lambda storage, loc: storage)
        return checkpoint['labels']

    def train_step(self, train_loader):
        """
        Model's training step.

        params:
            <DataLoader> train_loader: Training dataset containing the features and labels, shuffle and batched appropriately.

        returns:
            <float> err: The training loss at that step
            <float> acc: The training accuracy at that step
        """

        # Set model in train mode
        self.model.train()

        # Initialize running error and running accuracy metrics
        running_error = 0.0
        running_correct = 0

        # Iterate through the data
        for features, labels in train_loader:
            # Here we permute because pretrained models in Pytorch require inputs of shape (batch_size, num_channels, width, height)
            features = features.permute(0, 3, 1, 2)

            # Load data to the desired computation device
            features = set_device(features, self.cuda)
            labels = set_device(labels, self.cuda)

            # Reset the optimizer's gradients
            self.optimizer.zero_grad()

            # Enable gradients for gradient descent
            with torch.set_grad_enabled(True):
                # Prediction
                pred = self.model(features)

                # Cross entropy loss
                loss = self.criterion(torch.log(pred), torch.max(labels, 1)[1])

                # Backprop
                loss.backward()

                # Optimizer step
                self.optimizer.step()

            # Labels for metrics
            _, pred_idx = torch.max(pred.data, 1)
            _, true_idx = torch.max(labels.data, 1)

            # Metrics calculations
            running_error += loss.item() * features.size(0)
            running_correct += (pred_idx == true_idx).float().sum()

        # Total error
        err = running_error / len(train_loader.dataset)
        
        # Total accuracy
        acc = 100 * running_correct / len(train_loader.dataset)

        return err, acc

    def validation_step(self, validation_loader):
        """
        Model's validation step.

        params:
            <DataLoader> validation_loader: Validation dataset containing the features and labels, shuffle and batched appropriately.

        returns:
            <float> err: The validation loss at that step
            <float> acc: The validation accuracy at that step
        """

        # Set model in evaluation mode
        self.model.eval()

        # Initialize running error and running accuracy metrics
        running_error = 0.0
        running_correct = 0

        # Iterate through the data
        for features, labels in validation_loader:
            features = features.permute(0, 3, 1, 2)

            # Load data to the desired computation device
            features = set_device(features, self.cuda)
            labels = set_device(labels, self.cuda)

            # Reset the optimizer's gradients
            self.optimizer.zero_grad()

            # Disable gradients as there is no training done in the validation step
            with torch.no_grad():
                # Prediction
                pred = self.model(features)

                # Cross-entropy loss
                loss = self.criterion(torch.log(pred), torch.max(labels, 1)[1])

            # Labels for metrics
            _, pred_idx = torch.max(pred.data, 1)
            _, true_idx = torch.max(labels.data, 1)

            # Metrics calculations
            running_error += loss.item() * features.size(0)
            running_correct += (pred_idx == true_idx).float().sum().item()

        # Total error
        err = running_error / len(validation_loader.dataset)
        
        # Total accuracy
        acc = 100 * running_correct / len(validation_loader.dataset)

        return err, acc

    def fit(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs: int,
        batch_size: int,
        resume: bool
    ):
        """
        The model's fit process.

        params:
            <array> x_train
            <array> y_train
            <array> x_test
            <array> y_test
            <int> epochs
            <int> batch_size
            <bool> save: Whether to save progress or not
            <bool> resume: Whether to resume training or not
        """

        if self.cuda == True:
            torch.backends.cudnn.benchmark = True

        # Mode choice
        if self.mode == "feature_extractor":
            train_loader = DataLoader(dataset=list(zip(
                x_train, y_train)), shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
            validation_loader = DataLoader(dataset=list(zip(
                x_test, y_test)), shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
        else:
            # We want to augment the data when in feature composition mode.
            train_transform = transforms.Compose([
                transforms.ToPILImage("RGB"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

            val_transform = transforms.Compose([
                transforms.ToPILImage("RGB"),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

            x_train_ds = augmented_data(x_train, train_transform)
            x_test_ds = augmented_data(x_test, val_transform)

            train_loader = DataLoader(dataset=list(
                zip(x_train_ds, y_train)), shuffle=True, batch_size=batch_size)
            validation_loader = DataLoader(dataset=list(
                zip(x_test_ds, y_test)), shuffle=True, batch_size=batch_size)

        start_epoch = 0
        # If the user chooses to resume
        if resume == True:

            # List of checkpoints
            ckpt_paths_list = []
            for i, ckpt_path in enumerate(os.listdir(self.ckpt_dir)):
                if self.mode == "feature_extractor":
                    if "feature_extractor" in ckpt_path:
                        print(f"{i + 1}) {ckpt_path}")
                        ckpt_paths_list.append(ckpt_path)
                else:
                    if "feature_composer" in ckpt_path:
                        print(f"{i + 1}) {ckpt_path}")
                        ckpt_paths_list.append(ckpt_path)

            # Check if there are any available checkpoints
            assert len(ckpt_paths_list > 0)

            # Prompt user for a choice
            ckpt_path_choice = -1
            while ckpt_path_choice > len(model_paths_list) or ckpt_path_choice < 1:
                ckpt_path_choice = int(input(
                    f"Which model would you like to load? [Number between 1 and {len(ckpt_paths_list)}]: "))

            ckpt_path = os.path.join(
                self.ckpt_dir, ckpt_paths_list[ckpt_path_choice - 1])

            # Load said checkpoint and it's elements.
            checkpoint = torch.load(self.ckpt_path)
            start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Training loop
            progress_bar = tqdm(range(start_epoch, epochs))
            progress_bar.set_description("Resuming training...")
            for epoch in progress_bar:
                train_loss, train_acc = self.train_step(train_loader)
                val_loss, val_acc = self.validation_step(validation_loader)

                # Regularization
                self.scheduler.step(val_loss)

                # Save condition
                if epochs >= 10:
                    if (epoch + 1) % (epochs // 10) == 0:
                        self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)
                else:
                    self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)

                progress_bar.set_description(
                    f"[Epoch {epoch + 1} stats]: train_loss = {train_loss} | train_acc = {train_acc}% | val_loss = {val_loss} | val_acc = {val_acc}%")

        else:
            progress_bar = tqdm(range(start_epoch, epochs))
            progress_bar.set_description("Starting training...")

            for epoch in progress_bar:
                train_loss, train_acc = self.train_step(train_loader)
                val_loss, val_acc = self.validation_step(validation_loader)
                
                # Regularization
                self.scheduler.step(val_loss)

                # Save condition
                if epochs >= 10:
                    if (epoch + 1) % (epochs // 10) == 0:
                        self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)
                else:
                    self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)

                progress_bar.set_description(
                    f"[Epoch {epoch + 1} stats]: train_loss = {train_loss:.2f} | train_acc = {train_acc:.2f}% | val_loss = {val_loss:.2f} | val_acc = {val_acc:.2f}%")

    def infer(self, input_data, use_labels=False):
        """
        The model's inference process.

        params:
            <array> input_data
            <bool> use_labels: Whether to output nicely, with a description of the labels, or not
        returns:
            <array> prediction
        """

        # Disable gradients. We're not training.
        with torch.no_grad():
            # Convert the input data into a tensor if needed
            if type(input_data) != torch.Tensor:
                input_data = torch.Tensor(input_data)

            # Reshape if appropriately.
            input_data = input_data.reshape(-1, 3, 224, 224)

            # Prediction
            output = self.model.cpu()(input_data).numpy()

            if use_labels == True:
                labels = self.load_labels_for_inference()
                labeled_output = {labels[output.argmax()]: output}
                return labeled_output
            else:
                return output

    def infer_using_pretrained_layers_without_last(self, features):
        """
        Used when extracting the features.

        params:
            <array> features
        returns:
            <array> pred: NxN array (doesn't use custom classification layer)
        """

        # Keep track of the last layer
        last_layer_to_restore = list(self.model.children())[-1]
        last_layer = list(self.model.children())[-1][:-1]

        # Remove last layer
        while type(last_layer[-1]) != nn.modules.linear.Linear:
            last_layer = last_layer[:-1]

        # Update the model architecture
        try:
            self.model.classifier = last_layer
        except:
            self.model.fc = last_layer

        # We'll use the CPU for computation here, as continuous GPU inference tends to give an "Out of memory" error.
        self.model = self.model.cpu()

        # Convert to tensor
        features = torch.Tensor(features)

        # Disable gradients
        with torch.no_grad():
            # Predict
            features = features.permute(0, 3, 1, 2)
            pred = self.model(features)
        
        # Convert to array
        pred = pred.numpy()

        # Restore layer
        try:
            self.model.classifier = last_layer_to_restore
        except:
            self.model.fc = last_layer_to_restore

        # Return prediction
        return pred
