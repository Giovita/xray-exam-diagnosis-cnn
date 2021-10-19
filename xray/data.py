from google.cloud.storage import bucket
import numpy as np
import pandas as pd
import os
from itertools import chain

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage

from xray.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, BUCKET_TRAIN_CSV_PATH


def get_data(labels_file: str, source = 'csv'):

    """
    Loads raw csv into pd.DataFrame from multiple sources.
    Souces can be: '.csv', 'gcp', other
    """

    if source == 'csv':
        df = pd.read_csv(labels_file)
    elif source == 'gcp':
        # Add Client() here
        client = storage.Client()
        path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
        df = pd.read_csv(path,)
    return df


# Load Data, Encoding y Unificar clases iguales
def process_df(dataframe: pd.DataFrame, cols, save_df_to_csv=False):
    """
    Process df to input into cnn pipeline
    """
    # Generacion columna conteo de enfermedades por imagen
    dataframe["count_diseases"] = dataframe["Finding Labels"].map(
        lambda x: len(x.split("|")))
    dataframe["Count_diseases"] = np.where(
        dataframe["Finding Labels"] == "No Finding", 0,
        dataframe["count_diseases"])

    # eliminar columna count_diseases
    dataframe.drop(columns="count_diseases", inplace=True)

    # Generacion columna enfermo/no_enfermo --> True es enfermo - False es No enfermo
    dataframe["Enfermo"] = np.where(dataframe["Count_diseases"] == 0, "False",
                                  "True")  # se utiliza en string por keras

    # Multiple encoding de las clases
    all_labels = np.unique(
        list(
            chain(*dataframe["Finding Labels"].map(
                lambda x: x.split("|")).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]

    for c_label in all_labels:
        if len(c_label) > 1:
            dataframe[c_label] = dataframe["Finding Labels"].map(
                lambda finding: 1.0 if c_label in finding else 0)

    # Unficar clases --> ejemplo: (Infiltration|Effusion') y (Effusion|Infiltration')
    # cambiar de float a integer
    dataframe[cols] = dataframe[cols].applymap(np.int64)
    # Que es cols?


    dataframe["Combined"] = dataframe[cols].values.tolist()
    dataframe["Fixed_Labels"] = dataframe["Combined"].apply(
        lambda x: "|".join([cols[i] for i, val in enumerate(x) if val == 1]))

    # borrar Finding Labels (ya existe el reemplazo), borrar la columna combined (no sirve)
    dataframe.drop(columns=["Combined", "Finding Labels"], inplace=True)

    index_mayor100 = list(dataframe[dataframe["Patient Age"] > 100].index)  # Remove outlier patients
    dataframe = dataframe.drop(index_mayor100)
    if save_df_to_csv:
        dataframe.to_csv("dataframe.csv", index=False)

    return dataframe


def filter_dataset_from_list(dataset: pd.DataFrame, filter_list: list,
                             filter_by_col: str):
    """
    Return a test/train dataset split from the given index list.
    - test_set_filter: .csv file with list
    - filter_list = file/df/list containing desired elements for train set
    - filter by column: column to filter by

    To be used if a pre-ordained test set is provided.
    """

    if isinstance(filter_list, str):
        values_to_filter = pd.read_csv(filter_list)
    else:
        values_to_filter = (
            filter_list  # So it can be providded as df or list from pipeline
        )

    bool_filter = dataset[filter_by_col].isin(
        values_to_filter[values_to_filter.column[0]])

    y_test = dataset[bool_filter]
    y_train = dataset[bool_filter.map(lambda x: not x)]

    return y_train, y_test


def split_df(
    dataset: pd.DataFrame,
    column_to_filter_by,
    train_val_test: tuple = None,
    split: float = 0.85,
    total_filter=0.1,
    ):
    """
    Reduce total dataset according to 'total_filter'
    Randomly split a df, by a given column in a `split` split.
    * By default will give a train_val split. If a tuple is provided as train_val_test,
    it wil split three-fold.
    ex: column_to_filter = 'Patient ID'

    Execution notice: Being split by Patient ID, the final dataset will not have
    precisely train_val_test
    """

    assert sum(train_val_test) == 1, 'train-val-test split must add up to 1'

    if not column_to_filter_by:
        patients = dataset
    else:
        patients = pd.DataFrame(dataset[column_to_filter_by].unique(),
                            columns=[column_to_filter_by])

    # Filtered patients
    reduced_patients = patients.sample(frac=total_filter, )

    # Reduce full DS
    reduced_ds = dataset[dataset[column_to_filter_by].isin(
        reduced_patients[column_to_filter_by])]

    # Set split guide if train_val_test or just train_val
    if train_val_test:
        split_guide = train_val_test
    else:
        split_guide = (split, 1 - split, 0)

    length = reduced_patients.shape[0]

    train_idx = int(length * split_guide[0])
    val_idx = int(train_idx + length * split_guide[1])
    test_idx = int(val_idx + length * split_guide[2])

    patients_shuffle = reduced_patients.sample(
        frac=1)  # Randomize patients list

    # Pick train, val and test guides
    patients_train = patients_shuffle[0:train_idx]
    patients_val = patients_shuffle[train_idx:val_idx]
    patients_test = patients_shuffle[val_idx:test_idx]

    # Select every record for the patients in each file.
    ds_train = reduced_ds[reduced_ds[column_to_filter_by].isin(
        patients_train[column_to_filter_by])]
    ds_val = reduced_ds[reduced_ds[column_to_filter_by].isin(
        patients_val[column_to_filter_by])]
    ds_test = reduced_ds[reduced_ds[column_to_filter_by].isin(
        patients_test[column_to_filter_by])]

    return ds_train, ds_val, ds_test


def build_generator(
    img_path: str,
    labels_df: pd.DataFrame,
    labels_list: list = None,
    index_col="Image Index",
    labels_col="Finding Labels",
    # return_labels_dict=False,
    train_set_batch_size=32,
    target_size=(224,224),
    binary_class=False,
    test_set=False,
    data_augment=False,
    **kwargs):

    """
    Loads X_data, images, from a directory with files, and assings labels
    according to the DF in labels_path.
    While loading, filters labels not inb 'labels' list
    - img_path: relative path to directory containing folders
    - labels_df: pd.DataFrame or relative path to file (.csv) containing labels. Must include filename
    - labels_list: list of labels to consider, filtering out every other label.
    """
    if data_augment:
        datagen = ImageDataGenerator(rescale=1 / 255, **kwargs)
    else:
        datagen = ImageDataGenerator(rescale=1 / 255, )

    if binary_class:
        class_mode = "binary"
    else:
        class_mode = "categorical"

    if test_set:
        batch_size = 1
        shuffle = False
    else:
        batch_size = train_set_batch_size
        shuffle = True

    generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=img_path,
        x_col=index_col,
        y_col=labels_col,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=target_size,  # according to multiclass classification algorithm
        classes=labels_list,  # Filters files that don't belong to these classes
    )

    return generator


import random
from glob import glob

import tensorflow as tf


def make_dataset(path,
                 batch_size,
                 filenames:list,
                 labels: list,
                 img_size: tuple = (224, 224),
                 classes_in_folders=False):
    """
    - path: root to image folders
    - batch_size: to iterate
    - filenames: nd.array with list of absolute paths (filenames), in same order as label_array
    - label_array: matching index as filenames
    """
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        return image

    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    if classes_in_folders:
        classes = os.listdir(path)
        filenames = glob(path + "/*/*")
        it = np.nditer(filenames, flags=['refs_ok', 'c_index'], )
        random.shuffle(filenames)
        for file in it:
            labels = [classes.index(name.split("/")[-2]) for name in filenames]
    else:
        filenames = filenames
        labels = labels

    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(
        parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)

    return ds


def get_data_from_gcp(filename: str, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    # path = fr"gs://{BUCKET_NAME}/{BUCKET_TRAIN_CSV_PATH}/{filename}"
    path = os.path.join('gs://',BUCKET_NAME, filename)
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    """Test script to test data utilities. """

    import os

    path_to_csv = "../../raw_data/full-dataset/"
    csv_file = "xray_df.csv"
    df = get_data(os.path.join(path_to_csv, csv_file))
    ds_train, ds_val, ds_test = split_df(dataset=df,
                                              column_to_filter_by='Patient ID',
                                              train_val_test=(0.65, 0.15,
                                                              0.15))

    print('train :', ds_train.shape)
    print('val :', ds_val.shape)
    print('test :', ds_test.shape)
