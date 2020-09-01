import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model

import os
import time
from datetime import datetime

class DeTraC_model(tf.keras.Model):
    def __init__(self, pretrained_model, mode: str, class_names = None):
        super(DeTraC_model, self).__init__()

        self.pretrained_model = pretrained_model
        self.mode = mode

        self._initial_epoch = 0
        self._use_for_extraction = False

        self.custom_weights = lambda shape, dtype = None: \
            tf.Variable(lambda: tf.random.normal(shape) * 0.0001)

        self.custom_biases = lambda shape, dtype = None: \
            tf.Variable(lambda: tf.random.normal(shape) * 0.0001 + 1)

        self.pretrained_layers = pretrained_model.layers[:-2]
        self.classification_layer = Dense(
            units = self.num_classes,
            activation = 'softmax',
            kernel_initializer = self.custom_weights,
            bias_initializer = self.custom_biases
        )

        if self.mode == "feature_extractor":
            self.pretrained_layers.trainable = False
            self.classification_layer.trainable = True
        else:
            self.pretrained_layers.trainable = True
            self.classification_layer.trainable = True
            self.class_names = class_names

    def call(self, x):
        x = self.pretrained_layers(x)
        if self._use_for_extraction == False:
            x = self.classification_layer(x)
        return x

class DeTraC_callback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(DeTraC_callback, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.model._initial_epoch = epoch

class Net(object):
    """
    The DeTraC model.
    """
    def __init__(
        self, 
        pretrained_model, 
        num_classes: int, 
        mode: str,
        class_names: list = None):

        self.mode = mode
        self.num_classes = num_classes
        self.class_names = class_names

        assert self.mode == "feature_extractor" or self.mode == "feature_composer"

        now = datetime.now()
        now = f'{str(now).split(" ")[0]}_{str(now).split(" ")[1]}'.split(".")[0].replace(':', "-")
        if self.mode == "feature_extractor":
            self.save_name = f"DeTraC_feature_extractor_{now}.h5"
            self.optimizer = SGD(
                learning_rate = 1e-4,
                momentum = 0.9,
                nesterov = False,
                decay = 1e-3
            )
            self.scheduler = ReduceLROnPlateau(
                factor = 0.9,
                patience = 3
            )
        else:
            assert len(class_names) == num_classes
            self.save_name = f"DeTraC_feature_composer_{now}.h5"
            self.optimizer = SGD(
                learning_rate = 1e-4,
                momentum = 0.95,
                nesterov = False,
                decay = 1e-4
            )
            self.scheduler = ReduceLROnPlateau(
                factor = 0.95,
                patience = 5
            )

        self.model = DeTraC_model(self.pretrained_model, mode = self.mode, class_names = self.class_names)

        self.model_dir = "../../../models/tf"
        self.model_path = os.path.join(self.model_dir, self.save_name)

        self.model.compile(
            optimizer = self.optimizer,
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )

        self.model.summary()

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

        # TODO: Augment data for feature composer

        model_paths_list = []
        if resume == True:
            for i, model_path in enumerate(os.listdir(self.model_dir)):
                if model_path.endswith(".h5"):
                    print(f"{i}) {model}")
                    model_paths_list.append(model)
            
            assert len(model_paths_list > 0)
            
            model_path_choice = -1
            while model_path_choice > len(model_paths_list) or model_path_choice < 1:
                model_path_choice = int(input(f"Which model would you like to load? [Number between 1 and {len(model_list)}]: "))

            model_path = os.path.join(self.model_dir, model_paths_list[model_path_choice - 1])

            load_model(
                filepath = model_path,
                compile = False
            )
            self.model.fit(
                x = x_train,
                y = y_train,
                batch_size = batch_size,
                epochs = epochs,
                validation_data = [x_test, y_test],
                shuffle = True,
                initial_epoch = self.model._initial_epoch,
                callbacks = [
                    self.scheduler, 
                    DeTraC_callback(model = self.model),
                    ModelCheckpoint(
                        filepath = self.model_path,
                        monitor = "val_acc",
                        verbose = 1,
                        save_freq = epochs // 10
                    )
                ]
            )
        else:
            self.model._initial_epoch = 0
            self.model.fit(
                x = x_train,
                y = y_train,
                batch_size = batch_size,
                epochs = epochs,
                validation_data = [x_test, y_test],
                shuffle = True,
                initial_epoch = self.model._initial_epoch,
                callbacks = [
                    self.scheduler, 
                    DeTraC_callback(model = self.model),
                    ModelCheckpoint(
                        filepath = self.model_path,
                        monitor = "val_acc",
                        verbose = 1,
                        save_freq = epochs // 10
                    )
                ]
            )

    def infer(self, input_data, use_labels):
        input_data = input_data.reshape(-1, 224, 224, 3)
        output = self.model.predict(input_data, verbose = 2)

        if use_labels == True:
            labels = self.model.class_names
            labeled_output = {labels[output.argmax()] : output}
        else:
            return output

    def infer_using_pretrained_layers_without_last(self, features):
        self.model._use_for_extraction = True
        return self.model.predict(features)