import numpy as np
import pandas as pd
import os
from glob import glob

import matplotlib.pyplot as plt
from itertools import chain

# import tensorflow

# import tensorflow.keras
# import sklearn
# import sklearn as sk

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import f1_score
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential #, Model
from tensorflow.keras.layers import (
    # GlobalAveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    # Conv2D,
    # MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    # LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import Accuracy, Precision, Recall, CategoricalAccuracy

from xray.params import (MLFLOW_URI,
                        EXPERIMENT_NAME,
                        BUCKET_NAME,
                        MODEL_VERSION,
                        MODEL_VERSION)

from datetime import datetime

class Trainer():
    """
    Implements methods needed for training a CNN, including stor
    """

    def __init__(self, gen_train, gen_val, category_type):
        """
        *Generators from tf.keras.ImageDataGenerator class, previously split in
        train, test, val
        - category_type: 'binary', 'multicategory', 'multilabel'
        """
        self.pipeline = None
        self.gen_train = gen_train
        self.gen_val = gen_val
        self.category_type = category_type
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """How to automatically scale to 0-1 when predicting?"""
        pass

    def build_cnn(
            self,  # Provides train and val generators
            input_shape,
            output_shape,
            dense_layer_geometry: tuple,
            output_activation=None,
            transfer_model=VGG16,
            dense_layer_activation="relu",
            dropout_layers=True,
            dropout_rate=0.2,
            first_units=512,  # drop in refactor

    ):
        """
        params:
        - input_shape: (size, size, channels) of input images --> (128, 128, 3) TYPE TUPLE
        - output_shape: number of output units in the last Dense layer -->
            binary = 1 , multiple = n_labels  TYPE INT
        - dense_layer_geometry: geometry of each dense_layer aded.
        - output_activation: example: 'sigmoid' or 'softmax' TYPE STR
        - model_name: imported model for Transfer Learning
        - dense_layers_activation: activation for hidden units of dense networkexample: 'relu' TYPE STR
        - droput_layers: if droput layers are included in final geom
        - dropout_rate: percentage of dropout  TYPE FLOAT
        - first_units: architecture of first hidden dense layer. DEPRECATED
        """

        # if len(input_shape) == 2:
        #     input_shape += (3,)

        # Transfer Learning Import
        vgg_model = transfer_model(include_top=False,
                            weights="imagenet",
                            input_shape=input_shape)
        vgg_model.trainable = False

        # Params for dense_layers architecture  -- DEPRECATED for tuple_sintax
        second_unit = first_units / 2
        third_unit = second_unit / 2

        # Build final layers
        model = Sequential()
        model.add(vgg_model)
        model.add(Flatten())

        for neurons in dense_layer_geometry:
            model.add(Dense(neurons, activation=dense_layer_activation))
            if dropout_layers:
                model.add(Dropout(dropout_rate))

        output_activ_dict = {'binary': 'sigmoid',
                             'multicategorical': 'softmax',
                             'multilabel': 'sigmoid'}


        if not output_activation and self.category_type == 'binary':
            output_activation = output_activ_dict.get(self.category_type)
        else:
            if output_activation != output_activ_dict.get(self.category_type):
                print("Intended output activation function inconsistent. Please check")

        model.add(Dense(output_shape, activation=output_activation))

        self.pipeline = model

        # return model
        # # model.add(Dropout(dropout_rate))  # Drop because unnecesary
        # model.add(Dense(first_units, activation=dense_layers_activation))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(second_unit, activation=dense_layers_activation))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(third_unit, activation=dense_layers_activation))
        # model.add(Dense(output_unit, activation=output_activation))
        # Metricas y optimizador

    def compile_model(self,
            loss=None,
            learning_rate=1e-4,  # Default smaller than tf.keras def for Transfer L.
            metrics=None
            ):

        """
        Compile model with given params.
        If loss and metrics are not provided,
        """

        optimizer = Adam(learning_rate=learning_rate)

        if not metrics:
            if self.category_type == 'binary':
                metrics = [Accuracy(), Precision(), Recall()]
            else:
                metrics = [Accuracy(), Precision(), Recall(), CategoricalAccuracy()]

        if not loss:
            loss_dict = {'binary': 'binary_crossentropy',
                    'multicategorical': 'multicategory_crossentropy',
                    'multilabel': 'binary_crossentropy'}
            loss = loss_dict[self.category_type]

        self.pipeline.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=metrics)


    def fit_model(self,
                  callback=None,
                  patience=10,
                  epochs=20 ):

        es = EarlyStopping(monitor="val_loss",
                            mode="min",
                            patience=patience,
                            restore_best_weights=True)

        checkpoint = ModelCheckpoint('best_weights.hdf5',
                        save_best_only=True,
                        verbose=2)

        adapt_lr = ReduceLROnPlateau(patience=5, verbose=1)

        if not callback:
            callbacks_list = [es, checkpoint, adapt_lr]
        else:
            callbacks_list = callback

        self.pipeline.fit(self.gen_train,
                            validation_data=self.gen_val,
                            epochs=epochs,
                            callbacks=callbacks_list)

    def run(self):
        self.build_cnn()
        self.compile_model()
        # self.mlflow_log_param("model", "Linear")
        self.pipeline.fit()

        print('Fitted')
        # return history

    def save_locally(self):
        """Save model in tf.keras default model"""
        self.pipeline.save_weights(
            'drive/MyDrive/Proyecto_Lewagon_Rayos_X/models/model_vgg_multilabel_1.h5'
        )
        if not os.path.join(os.getcwd(), 'models'):
            os.mkdir('models')

        if self.experiment_name:
            name = self.experiment_name
        else:
            name = f"{self.pipeline.layers[0].name}_{f'{datetime.now()}'.replace(' ', '_')}"
            model_dir = os.path.join(os.path.join(os.getcwd(),
                                              'models', self.experiment_name))


# trainer should be instanciated with everything inherent to the model:
# parameters set in its methods, may update atributes, that default with None when
# instanciated


"""
TODO

refactor so every input is given on instanciation (except fit), add
doc to every method
"""
