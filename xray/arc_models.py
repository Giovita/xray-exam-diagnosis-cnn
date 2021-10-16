import numpy as np
import pandas as pd
import os
from glob import glob

import matplotlib.pyplot as plt
from itertools import chain

import tensorflow

# import keras
import tensorflow.keras
import sklearn
import sklearn as sk

from sklearn.model_selection import train_test_split


from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.metrics import Precision, Recall, Accuracy
<<<<<<< HEAD


# Load Data, Encoding y Unificar clases iguales
def load_and_process(data_folder, archivo, cols):

    # lectura del dataframe
    xray_df = pd.read_csv(data_folder + archivo)

    # Generacion columna conteo de enfermedades por imagen
    xray_df["count_diseases"] = xray_df["Finding Labels"].map(
        lambda x: len(x.split("|"))
    )
    xray_df["Count_diseases"] = np.where(
        xray_df["Finding Labels"] == "No Finding", 0, xray_df["count_diseases"]
    )

    # eliminar columna count_diseases
    xray_df.drop(columns="count_diseases", inplace=True)

    # Generacion columna enfermo/no_enfermo --> True es enfermo - False es No enfermo
    xray_df["Enfermo"] = np.where(
        xray_df["Count_diseases"] == 0, "False", "True"
    )  # se utiliza en string por keras

    # Multiple encoding de las clases
    all_labels = np.unique(
        list(chain(*xray_df["Finding Labels"].map(lambda x: x.split("|")).tolist()))
    )
    all_labels = [x for x in all_labels if len(x) > 0]

    for c_label in all_labels:
        if len(c_label) > 1:
            xray_df[c_label] = xray_df["Finding Labels"].map(
                lambda finding: 1.0 if c_label in finding else 0
            )

    # Unficar clases --> ejemplo: (Infiltration|Effusion') y (Effusion|Infiltration')
    # cambiar de float a integer
    xray_df[cols] = xray_df[cols].applymap(np.int64)

    xray_df["Combined"] = xray_df[cols].values.tolist()
    xray_df["Fixed_Labels"] = xray_df["Combined"].apply(
        lambda x: "|".join([cols[i] for i, val in enumerate(x) if val == 1])
    )

    # borrar Finding Labels (ya existe el reemplazo), borrar la columna combined (no sirve)
    xray_df.drop(columns=["Combined", "Finding Labels"], inplace=True)

    # eliminar pacientes sin sentido
    index_mayor100 = list(xray_df[xray_df["Patient Age"] > 100].index)
    xray_df = xray_df.drop(index_mayor100)

    xray_df.to_csv("xray_df.csv", index=False)

    return xray_df
=======
>>>>>>> 51d9592 (Refactor Data Pipeline, move  to data.py)


# aca debe llamar otra funcion que genera el path para cada imagen


# aca debe llamar otra funcion para la augmentation y el sample con el que vamos a entrenar
# val_gen debe ser generador


# llamar al modelo  y entrenar
def build_model(
    train_gen,
    val_gen,
    input_shape,
    loss,
    output_unit,
    output_activation,
    epochs=20,
    layers_activation="relu",
    dropout_rate=0.2,
    model_name=VGG16,
    first_units=512,
    learning_rate=1e-3,
    patience=10,
):
    """
    params:
    input_shape: (size, size, channels) --> (128, 128, 3) TYPE TUPLE
    layers_activation: example: 'relu' TYPE STR
    output_activation: example: 'sigmoid' or 'softmax' TYPE STR
    first_units: number of units in the first Dense layer TYPE INT
    output_unit: number of output units in the last Dense layer --> binary = 1 , multiple = n_labels  TYPE INT
    dropout_rate: percentage of dropout  TYPE FLOAT
    loss : 'binary_crossentropy' or 'categorical_crossentropy'
    """

    vgg_model = model_name(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    vgg_model.trainable = False

    second_unit = first_units / 2
    third_unit = second_unit / 2

    # CNN deep learning layers

    model = Sequential()

    model.add(vgg_model)

    model.add(Flatten())

    model.add(Dropout(dropout_rate))

    model.add(Dense(first_units, activation=layers_activation))

    model.add(Dropout(dropout_rate))

    model.add(Dense(second_unit, activation=layers_activation))

    model.add(Dropout(dropout_rate))

    model.add(Dense(third_unit, activation=layers_activation))

    model.add(Dense(output_unit, activation=output_activation))

    # Metricas y optimizador
    optimizer = Adam(learning_rate=learning_rate)

    loss = loss

    metrics = [Accuracy(), Precision(), Recall()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    early = EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
    )

    callbacks_list = [early]

    valX, valY = val_gen.next()

    history = model.fit(
        train_gen, validation_data=(valX, valY), epochs=epochs, callbacks=callbacks_list
    )

    return history


if __name__ == "__main__":

    ## Carpeta de los datos
    data_folder = "/content/drive/MyDrive/Proyecto_Lewagon_Rayos_X"
    ## importar DF
    archivo = "/Data_Entry_2017.csv"

    # columnas clases
    cols = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]

    xray_df = load_and_process(data_folder, archivo, cols)

    #  Size
    img_size = (128, 128)
    input_shape = img_size + (3,)

    #  epocas
    epochs = 30
