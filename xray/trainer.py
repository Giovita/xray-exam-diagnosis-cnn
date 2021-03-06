# from http.client import parse_headers
import os
from datetime import datetime
from pathlib import Path

import mlflow
from google.cloud import storage
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from tensorflow.keras import applications
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import AUC, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from xray.params import (
    BASE_MODEL_FOLDER,
    BUCKET_NAME,
    CHECKPOINT_FOLDER,
    EXPERIMENT_NAME,
    GCP_MODEL_STORAGE_LOCATION,
    MLFLOW_URI,
    PATH_TO_LOCAL_MODEL,
)


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

        self.TRANSFER_CNN = {
            "VGG16": applications.VGG16,
            "densenet": applications.DenseNet121,
            "ResNet50": applications.ResNet50,
            "Xception": applications.Xception,
            "InceptionV3": applications.InceptionV3,
        }

        # self.TRANSFER_CNN['VGG16']( )

        self.data_split = None
        self.train_obs = None
        self.train_val_test = None

        ## Compile attributes: modified at model compile
        self.base_arch = None
        self.dense_layer_num = None
        self.output_activation = None
        self.input_shape = None
        self.dense_layer_geom = None

        # Data loading and saving attrs
        self.filename = None  # Compile when save_model
        self.model_dir = os.path.join(BASE_MODEL_FOLDER, self.category_type)  # Relative route from root to model
        self.checkpoint_path = None
        self.experiment_name = EXPERIMENT_NAME  # For MlFlow logging
        self.save_local_dir = os.path.join(self.model_dir)
        self.save_gcp_dir = os.path.join(BUCKET_NAME, self.model_dir)

        self.data_augment = None

    def build_cnn(
        self,  # Provides train and val generators
        input_shape: tuple,
        dense_layer_geometry: tuple,
        output_shape: int = None,
        output_activation=None,
        transfer_model=applications.VGG16,
        dense_layer_activation="relu",
        dropout_layers=True,
        dropout_rate=0.2,
    ):
        """
        ***output_shape must be set to num_categories if not 'binary' classif
        params:
        - input_shape: (size, size, channels) of input images --> (224, 224, 3) TYPE TUPLE
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

        base_model = applications.VGG16(
            # base_model = applications.mobilenet.MobileNet(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape,
        )
        base_model.trainable = False

        self.base_arch = base_model.name

        # Build final layers
        model = Sequential()
        model.add(Rescaling(1.0 / 255))  # Rescaling(1./255, input_shape=input_shape)
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

        if self.category_type == "binary":
            output_shape = 1

        if not output_activation:  # and self.category_type == "binary":
            output_activation = output_activ_dict.get(self.category_type)
        else:
            if output_activation != output_activ_dict.get(self.category_type):
                print("Intended output activation function inconsistent. Please check")

        model.add(Dense(output_shape, activation=output_activation))

        # Set instance attributes

        self.output_activation = output_activation
        self.pipeline = model
        # self.model_dir = f"{self.base_arch}/{self.category_type}"
        # self.model_dir = os.path.join(, self.category_type)

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
                metrics = ["accuracy", Precision(), Recall(), AUC()]
            else:
                metrics = [
                    "accuracy",
                    Precision(),
                    Recall(),
                    AUC(),
                    CategoricalAccuracy(),
                ]

        if not loss:
            loss_dict = {
                "binary": "binary_crossentropy",
                "multicategorical": "multicategory_crossentropy",
                "multilabel": "binary_crossentropy",
            }
            loss = loss_dict[self.category_type]

        self.pipeline.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        # self.set_experiment_name(
        #     f"{EXPERIMENT_NAME}_{self.base_arch}_\
        #                                 {f'{datetime.now()}'.replace(' ', '_')}"
        # )

        self.filename = os.path.join(self.base_arch, str(datetime.now()).replace(" ", "_"))

        self.mlflow_log_param("filename", self.filename)

        params = {
            "base_arch": self.base_arch,
            "dense_layer_num": self.dense_layer_num,
            "prediction_type": self.category_type,
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
        self,
        callback=None,
        patience=10,
        epochs=20,
        steps_per_epoch=None,
        restart=False,
        **kwargs,
    ):

        es = EarlyStopping(monitor="val_recall", mode="min", patience=patience, restore_best_weights=True)

        # if not self.checkpoint_path:
        #     self.checkpoint_path = (
        #         f"{self.model_dir}/{self.experiment_name}/checkpoint/best_weights.hdf5"
        #     )
        #     print(f"Saved model in {self.model_dir}/{self.experiment_name}")
        # else:
        self.checkpoint_path = os.path.join(self.model_dir, "checkpoints")
        # self.pipeline.load_weights(os.path.join(self.checkpoint_path, 'best_weights.hdf5'))

        checkpoint = ModelCheckpoint(
            self.checkpoint_path,
            save_best_only=True,
            verbose=2,
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

        # self.mlflow_log_metric(history.history.keys(), history.history.values())

        # print(history.history)

        return history

    def run(self):
        self.build_cnn()
        self.compile_model()
        history = self.fit_model()

        print("Fitted")
        return history

    def evaluate_model(self, gen_test, steps=None, return_dict=True, **kwargs):
        metrics = self.pipeline.evaluate(gen_test, steps=steps, return_dict=return_dict, **kwargs)
        """
        When providing an infinite dataset, you must specify the number of steps
        to run (if you did not intend to create an infinite dataset, make sure to
        not call `repeat()` on the dataset).
        """
        metric_key = self.pipeline.metrics_names

        for k, v in metrics.items():
            self.mlflow_log_metric(k, v)

        # print(metrics)
        # print(metric_key)

        return metrics

    # def predict_xray(self, x):
    #     """Predict disease from xray img."""

    #     img = PIL.Image.open(x)
    #     img = image.resize(img, size=self.input_shape)
    #     img = image.grayscale_to_rgb(img)
    #     prediction = self.pipeline.predict(img)

    #     return prediction

    ### Modify Params

    def add_base_model(self, model_name: str, model):
        """Add a new tf.keras.application base model into the class."""

        self.TRANSFER_CNN[model_name] = model

    def set_experiment_name(self, experiment_name):
        """defines the experiment name for MLFlow"""
        self.experiment_name = experiment_name

    ### Save and Load utilities

    def save_locally(self, model_folder: str = None):
        """Save model in tf.keras default model"""

        if not model_folder:
            model_folder = self.model_dir

        if not os.path.join(os.getcwd(), model_folder):
            os.mkdir(model_folder)

        # self.model_dir = os.path.join(os.path.join(os.getcwd(), model_folder))
        # self.filename = f"{self.experiment_name}.h5"
        self.pipeline.save(os.path.join(self.model_dir, f"{self.filename}.h5"), save_format="h5")

    def upload_model_to_gcp(self, rm=True):
        """Upload current model to gcp location"""
        client = storage.Client().bucket(BUCKET_NAME)

        destination = os.path.join(self.model_dir, self.filename)
        blob = client.blob(destination)

        local_path = os.path.join(os.getcwd(), self.model_dir, self.filename)

        blob.upload_from_filename(f"{local_path}.h5")

        # if os.path.isfile(local_path):

        print(
            f"=> {self.filename} uploaded to bucket {BUCKET_NAME} inside {GCP_MODEL_STORAGE_LOCATION}/{self.model_dir}"
        )

        if rm:
            os.remove(os.path.join(self.model_dir, self.filename))

    def load_model_from_gcp(self, origin_model_dir, filename, dest_dir, **kwargs):
        """
        Upload current model to gcp location

        -model_dir: route from root (BUCKET) upto base geometry.
            So for instance, `model_dir = 'models/multilabelbase_geometry/' or ''models/binarybase_geometry/'
        - filename = used as 'date_time of model compiling'
        - dest_dir: relative path of destination folder.
        """

        client = storage.Client().bucket(BUCKET_NAME)

        origin = os.path.join(origin_model_dir, filename)
        blob = client.blob(origin)

        Path(dest_dir).mkdir(parents=True, exist_ok=True)

        filepath = os.path.join(os.getcwd(), dest_dir, filename)

        if filename.split(".")[-1] != "h5":
            blob.download_to_filename(f"{filepath}.h5")
        else:
            blob.download_to_filename(filepath)

        # if os.path.isfile(local_path):
        if filename.split(".")[-1] != "h5":
            self.pipeline = load_model(f"{filepath}.h5")
        else:
            self.pipeline = load_model(f"{filepath}.h5")

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        self.save_locally()
        print("saved model locally")

        # Implement here
        self.upload_model_to_gcp(rm=False)
        print(f"uploaded model to gcp cloud storage under \n => {BUCKET_NAME}")

    def load_model(self, model_folder: str = PATH_TO_LOCAL_MODEL, **kwargs):
        """Save model in tf.keras default model"""
        # if not os.path.join(os.getcwd(), model_folder):
        #     os.mkdir(model_folder)

        self.pipeline.load_model(os.path.join(self.model_dir, self.filename), **kwargs)

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
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    import math
    import os
    from glob import glob

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MultiLabelBinarizer

    from xray import data, params, trainer, utils

    # Some Parameters
    filename = "xray_df.csv"
    img_size = (224, 224)
    job_type = "multilabel"
    split = (0.65, 0.175, 0.175)
    data_filter = 0.05
    cnn_geometry = (1024, 512, 256)
    dropout_layer = False
    batch_size = 32
    epochs = 1

    print(f"Start building and training CNN for {job_type}.")

    print("Set Parameters")

    # Load data
    path_to_png = params.GCP_IMAGE_BUCKET
    df = data.get_data_from_gcp(filename)

    print("Loaded Training Data")

    # Train multilabel for sick people. Modify if binary class
    df = df[df["Fixed_Labels"] != "No Finding"]

    print(f"Total {len(df)} files loaded")

    # Small data ETL
    df["path"] = df.path.map(lambda x: "/".join(x.split("/")[-3:]))  # Relative paths to file loc
    df.path = df.path.map(lambda x: os.path.join(params.GCP_IMAGE_BUCKET, x))  # Absolute path in GCP
    df["labels"] = df["Fixed_Labels"].map(lambda x: x.split("|"))  # 'cat_col' not working

    # OneHot Encode multilabel
    mlb = MultiLabelBinarizer().fit(df.labels)
    y = mlb.transform(df.labels).astype("int16")
    y = y.tolist()

    print("Finished preprocessing")

    # Train, val, test split
    df_train, df_val, df_test = data.split_df(df, "Patient ID", split, total_filter=data_filter)
    df_train = df_train.path.to_list()
    df_val = df_val.path.to_list()
    df_test = df_test.path.to_list()

    print(f"Finished reducing and splitting Data. Kept {len(df)*data_filter} records")

    # Make tf.data.Dataset
    ds_train = data.make_dataset(path_to_png, 32, df_train, y)
    ds_val = data.make_dataset(path_to_png, 32, df_val, y)
    ds_test = data.make_dataset(path_to_png, 32, df_test, y, test_set=True)

    classes_dict = pd.DataFrame(mlb.classes_).to_dict()[0]
    classes = mlb.classes_

    print(f"Finished making tf.data.Datasets with classes: {classes}")

    # Trainer()
    model = trainer.Trainer(ds_train, ds_val, job_type)

    print("Instanciated Trainer()")

    model.build_cnn(
        input_shape=img_size,
        output_shape=len(classes),
        dense_layer_geometry=(1024, 512, 256),
        dropout_layers=dropout_layer,
        dropout_rate=0.25,
    )

    print(f"Built CNN with {model.base_arch} base model.")

    model.compile_model()

    print(f"Finished Compiling model. ")

    training_images = len(df_train)
    steps_per_epoch = math.ceil(training_images / batch_size)

    validation_images = len(df_val)
    validation_steps = math.ceil(validation_images / batch_size)

    test_images = len(df_test)
    test_steps = math.ceil(test_images / batch_size)

    print(f"Start model fitting for {epochs} epochs")

    history = model.fit_model(
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    print(f"Finished training with {history.history} results.")

    # model.save_locally()

    print("Evaluating performance")
    results = model.evaluate_model(
        ds_test,
    )  # steps=ds_test)
    # print(f"Results: {results.histoy}")

    # model.save_model()

    model.upload_model_to_gcp()

    print("Saved model")

# trainer should be instanciated with everything inherent to the model:
# parameters set in its methods, may update atributes, that default with None when
# instanciated
# """
# TODO

# refactor so every input is given on instanciation (except fit), add
# doc to every method  --> is it best design?

# """
