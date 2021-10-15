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

    generator = datagen.flow_from_dataframe(
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

    return generator

# Filter % of dataset --> filter in input DF



def filter_dataset_from_list(dataset: pd.DataFrame, filter_list: list,
                              filter_by_col: str):
    """
    Return a test/train dataset split from the given index list.
    - test_set_filter: .csv file with list
    - filter_list = file/df/list containing desired elements for train set
    - filter by column: column to filter by
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


def split_df(dataset: pd.DataFrame, column_to_filter, split: float = 0.85, total_filter=0.1):
    """
    Reduce total dataset according to 'total_filter'

    Randomly split a df, by a given column in a `split` split.

    ex: column_to_filter = 'Patient ID'
    """

    patients = pd.DataFrame(dataset[column_to_filter].unique(),
                            columns=[column_to_filter])

    reduced_ds = patients.sample(frac=total_filter, )

    train_guide = reduced_ds.sample(frac=split, random_state=42)

    bool_filter = dataset[column_to_filter].isin(train_guide[column_to_filter])

    y_train = dataset[bool_filter]
    y_val = dataset[bool_filter.map(lambda x: not x)]

    return y_train, y_val


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

# 0. Get data. : use data paths relative. How to implement in cloud?
# 1. Split train - test
# 2. Split val - train
# 3. Generate generators, one for each set.
# 3.1. Binary, apply to image index, and binary labels
# 3.2. Multicategory, apply to image index, and Processed labels.


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
