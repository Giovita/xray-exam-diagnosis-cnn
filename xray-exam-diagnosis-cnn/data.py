import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def get_data(
    img_path: str,
    labels_file: str,
    labels: list = None,
    index_col='Image Index',
    labels_col='Finding Labels',
    return_labels_dict=False,
    binary_class=False,
    test_set = False
    ):
    """
    Loads X_data, images, from a directory with files, and assings labels
    according to the DF in labels_path.
    While loading, filters labels not inb 'labels' list
    - img_path: relative path to directory containing folders
    - labels_file: relative path to file (.csv) containing labels. Must include filename
    - labels: list of labels to consider, filtering out every other label.
    """

    # Load y_labels
    y = pd.read_csv(labels_file)
    y = y[['Image Index', 'Finding Labels']]

    datagen = ImageDataGenerator(rescale=1/255, )

    if binary_class:
        class_mode = 'binary'
    else:
        class_mode = 'categorical'

    if test_set:
        batch_size = 1
        shuffle = False
    else:
        batch_size = 32
        shuffle = True

    train_generator = datagen.flow_from_dataframe(
        dataframe=y,
        directory=img_path,
        x_col=index_col,
        y_col=labels_col,
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
        target_size=(228, 228),  # according to multiclass classification algorithm
        classes= labels  # Filters files that don't belong to these classes
    )

    return train_generator



# def get_binary_data(
#     img_path: str,
#     labels_file: str,
#     labels: list = None,
#     index_col='Image Index',
#     labels_col='Finding Labels',
#     return_labels_dict=False,
#     ):
#     """
#     Loads X_data, images, from a directory with files, and assings labels
#     according to the DF in labels_path.
#     While loading, filters labels not in 'labels' list
#     - img_path: relative path to directory containing folders
#     - labels_file: relative path to file (.csv) containing labels. Must include filename
#     - labels: list of labels to consider, filtering out every other label.
#     """

#     # Load y_labels
#     y = pd.read_csv(labels_file)
#     y['labels_binary'] = np.where(y['Finding Labels'] != 'No Finding',
#                                   'Disease', y['Finding Labels'])
#     y = y[['Image Index', 'labels_binary']]

#     datagen = ImageDataGenerator(rescale=1 / 255, )

#     train_generator = datagen.flow_from_dataframe(
#         dataframe=y,
#         directory=img_path,
#         x_col=index_col,
#         y_col=labels_col,
#         batch_size=32,
#         shuffle=True,
#         class_mode="binary",
#         target_size=(228,
#                      228),  # according to multiclass classification algorithm
#         classes= labels  # Filters files that don't belong to these classes
#     )

#     if return_labels_dict:
#         return y, train_generator, train_generator.class_indices
#     else:
#         return train_generator


def filter_datraset_from_list(dataset: pd.DataFrame, filter_list, filter_by_col: str):
    """
    Return a test/train dataset split from the given index list.
    -test_set_filter: .csv file with list
    """
    if isinstance(filter_list, str):
        values_to_filter = pd.read_csv(filter_list)
    else:
        values_to_filter = filter_list  # So it can be providded as df or list from pipeline

    bool_filter = dataset[filter_by_col].isin(
        values_to_filter[values_to_filter.column[0]])


    y_test = dataset[bool_filter]
    y_train = dataset[bool_filter.map(lambda x: not x)]

    return y_train, y_test


if __name__ == '__main__':
    import os

    # Paths to files
    path_from_root = "raw_data/sample-data/"
    path_to_root = "../"
    labels_path = os.path.join(path_to_root, path_from_root, "sample_labels.csv")
    img_path = os.path.join(path_to_root, path_from_root, "images")
    # labels = 'Data_Entry_2017.csv'  # FUll dataset labels
    pass
    # y, X = get_data(img_path, labels_path,  )
