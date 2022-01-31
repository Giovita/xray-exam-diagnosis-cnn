from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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

    vgg_model = model_name(include_top=False, weights="imagenet", input_shape=input_shape)

    vgg_model.trainable = False

    second_unit = first_units / 2
    third_unit = second_unit / 2

    # CNN deep learning layers
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(first_units, activation=layers_activation))
    model.add(Dense(second_unit, activation=layers_activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(third_unit, activation=layers_activation))
    model.add(Dense(output_unit, activation=output_activation))

    #  Metrics and optimizer
    optimizer = Adam(learning_rate=learning_rate)
    loss = loss
    metrics = [Accuracy(), Precision(), Recall()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True)
    callbacks_list = [early]
    valX, valY = val_gen.next()
    history = model.fit(train_gen, validation_data=(valX, valY), epochs=epochs, callbacks=callbacks_list)

    return history


if __name__ == "__main__":

    ## Carpeta de los datos
    data_folder = "/content/drive/MyDrive/Proyecto_Lewagon_Rayos_X"
    archivo = "/Data_Entry_2017.csv"

    # Classes (columns for OHE DF)
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
