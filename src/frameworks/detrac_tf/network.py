import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import save_model, load_model

import os
import time
from datetime import datetime

# Callback used for saving
class DeTraC_callback(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        model: Sequential, 
        num_epochs: int, 
        filepath: str
    ):
        super(DeTraC_callback, self).__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10:
            save_model(
                model=self.model,
                filepath=self.filepath,
                save_format='tf'
            )

    def on_epoch_begin(self, epoch, logs=None):
        self.model._initial_epoch = epoch

# The network
class Net(object):
    """
    The DeTraC model.
    """

    def __init__(
        self,
        pretrained_model: Model,
        num_classes: int,
        mode: str,
        model_dir: str,
        class_names: list = []
    ):

        """
        params:
            <Sequential> pretrained_model: VGG, AlexNet or whatever other ImageNet pretrained model is chosen
            <int> num_classes
            <string> mode: The DeTraC model contains 2 modes which are used depending on the case:
                                - feature_extractor: used in the first phase of computation, where the pretrained model is used to extract the main features from the dataset
                                - feature_composer: used in the last phase of computation, where the model is now training on the composed images, using the extracted features and clustering them.
            <list> class_names
        """

        self.pretrained_model = pretrained_model
        self.mode = mode
        self.num_classes = num_classes
        self.class_names = class_names
        self.model_dir = model_dir

        # Check if model directory exists
        assert os.path.exists(self.model_dir)

        # Check whether mode is correct
        assert self.mode == "feature_extractor" or self.mode == "feature_composer"

        now = datetime.now()
        now = f'{str(now).split(" ")[0]}_{str(now).split(" ")[1]}'.split(".")[0].replace(':', "-")

        # Initialize custom weights
        self.custom_weights = lambda shape, dtype = None: \
            tf.Variable(lambda: tf.random.normal(shape) * 0.0001)

        # Initialize custom biases
        self.custom_biases = lambda shape, dtype = None: \
            tf.Variable(lambda: tf.random.normal(shape) * 0.0001 + 1)

        # Pretrained layers
        self.pretrained_layers = Sequential(pretrained_model.layers[:-2])
        
        # Custom classification layer
        self.classification_layer = Dense(
            units=self.num_classes,
            activation='softmax',
            kernel_initializer=self.custom_weights,
            bias_initializer=self.custom_biases
        )

        # Set the save path, freeze or unfreeze the gradients based on the mode and define appropriate optimizers and schedulers.
        # Feature extractor => Freeze all gradients except the custom classification layer
        # Feature composer => Unfreeze / Activate all gradients
        if self.mode == "feature_extractor":
            self.pretrained_layers.trainable = False
            self.classification_layer.trainable = True
            self.save_name = f"DeTraC_feature_extractor_{now}"
            self.optimizer = SGD(
                learning_rate=1e-4,
                momentum=0.9,
                nesterov=False,
                decay=1e-3
            )
            self.scheduler = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.9,
                patience=3
            )
        else:
            self.pretrained_layers.trainable = True
            self.classification_layer.trainable = True
            assert len(class_names) == num_classes
            self.save_name = f"DeTraC_feature_composer_{now}"
            self.optimizer = SGD(
                learning_rate=1e-4,
                momentum=0.95,
                nesterov=False,
                decay=1e-4
            )
            self.scheduler = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.95,
                patience=5
            )
            
        # Instantiate model
        self.model = Sequential([self.pretrained_layers, self.classification_layer])
        self.model_path = os.path.join(self.model_dir, self.save_name)

        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
        resume: bool
    ):
        """
        Training function for the DeTraC model.

        params:
            <array> x_train
            <array> y_train
            <array> x_test
            <array> y_test
            <int> epochs
            <int> batch_size 
            <bool> resume
        """

        # If the feature composer is being used, augment the data
        if self.mode == "feature_composer":
            # Instantiate an image data generator
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_std_normalization = True,
                horizontal_flip = True
            )
            datagen.fit(x_train)

        # Instantiate the custom DeTraC callback
        custom_callback = DeTraC_callback(
            model=self.model,
            num_epochs=epochs,
            filepath=self.model_path
        )

        # If user wants to resume
        if resume == True:
            # List of models
            model_paths_list = []
            for i, model_path in enumerate(os.listdir(self.model_dir)):
                if self.mode == "feature_extractor":
                    if "feature_extractor" in model_path:
                        print(f"{i + 1}) {model_path}")
                        model_paths_list.append(model_path)
                else:
                    if "feature_composer" in model_path:
                        print(f"{i + 1}) {model_path}")
                        model_paths_list.append(model_path)

            # Check if there are available models
            assert len(model_paths_list) > 0

            # Prompt the user for a choice
            model_path_choice = -1
            while model_path_choice > len(model_paths_list) or model_path_choice < 1:
                model_path_choice = int(input(
                    f"Which model would you like to load? [Number between 1 and {len(model_paths_list)}]: "))

            model_path = os.path.join(self.model_dir, model_paths_list[model_path_choice - 1])

            # Load model
            load_model(
                filepath=model_path,
                compile=False
            )
            
            # Train.
            # If the feature extractor is selected, train normally
            if self.mode == "feature_extractor":
                self.model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    shuffle=True,
                    verbose=1,
                    callbacks=[
                        self.scheduler,
                        custom_callback
                    ]
                )
            # Otherwise, train on augmented data
            else:
                self.model.fit(
                    x=datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    steps_per_epoch=len(x_train) // batch_size,
                    shuffle=True,
                    verbose=1,
                    callbacks=[
                        self.scheduler,
                        custom_callback
                    ]
                )
            
        # If the user doesn't want to resume, train normally
        else:
            if self.mode == "feature_extractor":
                self.model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    shuffle=True,
                    verbose=1,
                    callbacks=[
                        self.scheduler,
                        custom_callback
                    ]
                )
            else:
                self.model.fit(
                    x=datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    steps_per_epoch=len(x_train) // batch_size,
                    shuffle=True,
                    verbose=1,
                    callbacks=[
                        self.scheduler,
                        custom_callback
                    ]
                )

    # Inference
    def infer(
        self, 
        input_data: np.ndarray, 
        use_labels: bool = False
    ) -> dict or np.ndarray:
        """
        Inference function.

        params:
            <array> input_data
            <bool> use_labels: Whether to output nicely, with a description of the labels, or not
        returns:
            <array> prediction
        """

        # Prediction
        output = self.model.predict(input_data)
        if use_labels == True:
            labels = self.class_names
            labeled_output = {labels[output.argmax()]: output}
            return labeled_output
        else:
            return output

    def infer_using_pretrained_layers_without_last(
        self, 
        features: np.ndarray
    ) -> np.ndarray:
        """
        Feature extractor's inference function.

        params:
            <array> features
        returns:
            <array> NxN array representing the features of an image
        """
        
        # Instantiate a sequential model
        extractor = Sequential()

        # Add all the pretrained layers to it
        for layer in self.model.layers[:-1]:
            extractor.add(layer)

        # Use the extractor to predict upon the input image
        output = extractor.predict(features)
        return output
