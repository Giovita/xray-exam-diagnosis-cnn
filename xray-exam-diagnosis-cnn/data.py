import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_data(img_path: str, labels_file: str, labels: list):
    """
    Loads X_data, images, from a directory with files, and assings labels
    according to the DF in labels_path.
    While loading, filters labels not in 'labels' list
    - img_path: relative path to directory containing folders
    - labels_file: relative path to file (.csv) containing labels. Must include filename
    - labels: list of labels to consider, filtering out every other label.
    """

    # Load y_labels
    y = pd.read_csv(labels_file)
    y = y[['Image Index', 'Finding Labels']]

    datagen = ImageDataGenerator(rescale=1 / 255, )

    train_generator = datagen.flow_from_dataframe(
        dataframe=y,
        directory=img_path,
        x_col="Image Index",
        y_col="Finding Labels",
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
        target_size=(228, 228),  # according to multiclass classification algorithm
        # classes= list(all_labels)  # Filters files that don't belong to these classes
    )

    return y, train_generator




def get_binary_data(img_path: str, labels_file: str, labels: list):
    """
    Loads X_data, images, from a directory with files, and assings labels
    according to the DF in labels_path.
    While loading, filters labels not in 'labels' list
    - img_path: relative path to directory containing folders
    - labels_file: relative path to file (.csv) containing labels. Must include filename
    - labels: list of labels to consider, filtering out every other label.
    """

    # Load y_labels
    y = pd.read_csv(labels_file)
    y['labels_binary'] = np.where(y['Finding Labels'] != 'No Finding',
                                  'Disease', y['Finding Labels'])
    y = y[['Image Index', 'labels_binary']]

    datagen = ImageDataGenerator(rescale=1 / 255, )

    train_generator = datagen.flow_from_dataframe(
        dataframe=y,
        directory=img_path,
        x_col="Image Index",
        y_col="labels_binary",
        batch_size=32,
        shuffle=True,
        class_mode="binary",
        target_size=(228,
                     228),  # according to multiclass classification algorithm
        # classes= list(all_labels)  # Filters files that don't belong to these classes
    )

    return y, train_generator


if __name__ == '__main__':
    import os

    # Paths to files
    path_from_root = "raw_data/sample-data/"
    path_to_root = "../"
    labels_path = os.path.join(path_to_root, path_from_root)
    img_path = os.path.join(path_to_root, path_from_root, "images")
    # labels = 'Data_Entry_2017.csv'  # FUll dataset labels
    labels = "sample_labels.csv"

    y, X = get_data(img_path, )
