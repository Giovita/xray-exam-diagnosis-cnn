import os
from datetime import datetime

from tensorflow.keras.applications import (
    VGG16,
    DenseNet121,
    ResNet50,
    Xception,
    InceptionV3,
)
from tensorflow.keras.models import Sequential  # , Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import (
    Accuracy,
    Precision,
    Recall,
    CategoricalAccuracy,
    AUC,
)
import PIL.Image
from tensorflow import image

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from xray.params import (
    GCP_MODEL_STORAGE_LOCATION,
    BUCKET_NAME,
    MLFLOW_URI,
    EXPERIMENT_NAME,
    GCP_IMAGE_BUCKET,
    MODEL_VERSION,
    GCP_MODEL_BUCKET,
    PATH_TO_LOCAL_MODEL,
)

from google.cloud import storage


class Trainer:
    """
    Implements methods needed for training a CNN.
    """

    MLFLOW_URI = MLFLOW_URI

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

        self.TRANSFER_CNN = {
            "VGG16": VGG16(),
            "densenet": DenseNet121,
            "ResNet50": ResNet50,
            "Xception": Xception,
            "InceptionV3": InceptionV3,
        }

        ## Compile attributes: modified at model compile
        self.base_arch = None
        self.dense_layer_num = None
        self.output_activation = None
        self.input_shape = None
        self.dense_layer_geom = None

        # Data loading and saving attrs
        self.filename = None  # Compile when save_model
        self.model_dir = None  # Relative route from root to model

    def set_experiment_name(self, experiment_name):
        """defines the experiment name for MLFlow"""
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

        # Transfer Learning Import
        if isinstance(transfer_model, str):
            transfer_model = self.TRANSFER_CNN.get(transfer_model)

        if len(input_shape) == 2:
            self.input_shape = input_shape + (3,)
        else:
            self.input_shape = input_shape

        base_model = VGG16(
            include_top=False, weights="imagenet", input_shape=self.input_shape
        )
        base_model.trainable = False

        # Build final layers
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())

        for neurons in dense_layer_geometry:
            model.add(Dense(neurons, activation=dense_layer_activation))
            if dropout_layers:
                model.add(Dropout(dropout_rate))

        self.dense_layer_num = len(dense_layer_geometry)
        self.dense_layer_geom = dense_layer_geometry

        output_activ_dict = {
            "binary": "sigmoid",
            "multicategorical": "softmax",
            "multilabel": "sigmoid",
        }

        if not output_activation:  # and self.category_type == "binary":
            output_activation = output_activ_dict.get(self.category_type)
        else:
            if output_activation != output_activ_dict.get(self.category_type):
                print("Intended output activation function inconsistent. Please check")

        self.output_activation = output_activation

        model.add(Dense(output_shape, activation=output_activation))

        self.pipeline = model

    def compile_model(
        self,
        loss=None,
        learning_rate=1e-4,  # Default smaller than tf.keras def for Transfer L.
        metrics=None,
        **kwargs,
    ):
        """
        Compile model with given params.
        If loss and metrics are not provided, default values depending on type of
        predictor are used
        """

        optimizer = Adam(learning_rate=learning_rate)

        if not metrics:
            if self.category_type == "binary":
                metrics = [Accuracy(), Precision(), Recall()]
            else:
                metrics = [Accuracy(), Precision(), Recall(), CategoricalAccuracy()]

        if not loss:
            loss_dict = {
                "binary": "binary_crossentropy",
                "multicategorical": "multicategory_crossentropy",
                "multilabel": "binary_crossentropy",
            }
            loss = loss_dict[self.category_type]

        self.pipeline.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        self.set_experiment_name(
            f"{EXPERIMENT_NAME}_{self.pipeline.layers[0].name}_\
                                        {f'{datetime.now()}'.replace(' ', '_')}"
        )

        params = {
            "base_arch": self.base_arch,
            "dense_layer_num": self.dense_layer_num,
            "output_activation": self.output_activation,
            "input_shape": self.input_shape,
            "dense_layer_geom": self.dense_layer_geom,
        }

        # params_value = [
        #     self.base_arch,
        #     self.dense_layer_num,
        #     self.output_activation,
        #     self.input_shape,
        #     self.dense_layer_geom,
        # ]
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def fit_model(
        self, callback=None, patience=10, epochs=20, steps_per_epoch=None, **kwargs
    ):

        es = EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            "best_weights.hdf5", save_best_only=True, verbose=2
        )

        adapt_lr = ReduceLROnPlateau(patience=5, verbose=1)

        if not callback:
            callbacks_list = [es, checkpoint, adapt_lr]
        else:
            callbacks_list = callback

        history = self.pipeline.fit(
            self.gen_train,
            validation_data=self.gen_val,
            epochs=epochs,
            callbacks=callbacks_list,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

        return history

    def run(self):
        self.build_cnn()
        self.compile_model()
        self.fit_model()

        print("Fitted")
        # return history

    def evaluate_model(self, gen_test, **kwargs):
        metric_values = self.pipeline.evaluate(
            gen_test, workers=4, use_multiprocessing=True, **kwargs
        )

        metric_key = self.pipeline.metrics_names

        self.mlflow_log_metric(metric_key, metric_values)

    def predict_xray(self, x):
        """Predict disease from xray img."""

        img = PIL.Image.open(x)
        img = image.resize(img, size=self.input_shape)
        img = image.grayscale_to_rgb(img)
        prediction = self.pipeline.predict(img)

        return prediction

    def add_base_model(self, model_name: str, model):
        """Add a new tf.keras.application base model into the class."""

        self.TRANSFER_CNN[model_name] = model

    ### Save and Load utilities

    def save_locally(self, model_folder: str = PATH_TO_LOCAL_MODEL):
        """Save model in tf.keras default model"""
        if not os.path.join(os.getcwd(), model_folder):
            os.mkdir(model_folder)

        self.model_dir = os.path.join(os.path.join(os.getcwd(), model_folder))
        self.filename = f"{self.experiment_name}.h5"
        self.pipeline.save(os.path.join(self.model_dir, self.filename))

    def upload_model_to_gcp(self, rm=False):
        """Upload current model to gcp location"""
        client = storage.Client().bucket(BUCKET_NAME)
        blob = client.blob(GCP_MODEL_STORAGE_LOCATION)
        blob.upload_from_filename(os.path.join(self.model_dir, self.filename))

        print(
            f"=> {self.filename} uploaded to bucket {BUCKET_NAME} inside {GCP_MODEL_STORAGE_LOCATION}",
            "green",
        )

        if rm:
            os.remove(self.filename)

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


# trainer should be instanciated with everything inherent to the model:
# parameters set in its methods, may update atributes, that default with None when
# instanciated
"""
TODO

refactor so every input is given on instanciation (except fit), add
doc to every method  --> is it best design?

"""
