import math
import os
from glob import glob

import pandas as pd
from xray import data, params, trainer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

if __name__ == "__main__":

    # Some Parameters
    filename = "xray_df.csv"
    img_size = (224, 224)
    job_type = "binary"
    split = (0.65, 0.175, 0.175)  # Train Val Test
    data_filter = 0.28
    cnn_geometry = (1024, 512, 256)
    dropout_layer = False
    dropout_rate = 0.2
    batch_size = 32
    epochs = 20
    # learning_rate = 0.001


    print(f"Start building and training CNN for {job_type}.")

    print("Set Parameters")

    # Load data
    path_to_png = params.GCP_IMAGE_BUCKET
    df = data.get_data_from_gcp(filename)

    print("Loaded Training Data")

    # Train multilabel for sick people. Modify if binary class
    # df = df[df["Fixed_Labels"] != "No Finding"]

    print(f"Total {len(df)} files loaded")

    # Small data ELT
    df["path"] = df.path.map(
        lambda x: "/".join(x.split("/")[-3:]))  # Relative paths to file loc
    df.path = df.path.map(lambda x: os.path.join(params.GCP_IMAGE_BUCKET, x)
                          )  # Absolute path in GCP
    # df["labels"] = df["Fixed_Labels"].map(
    #     lambda x: x.split("|"))  # 'cat_col' not working
    df['labels'] = df['Enfermo']

    # OneHot Encode multilabel
    # mlb = MultiLabelBinarizer().fit(df.labels)
    # y = mlb.transform(df.labels).astype("int16")
    # y = y.tolist()

    # Binary encode binary labels
    mlb = LabelEncoder().fit(df.labels)
    y = mlb.transform(df.labels).astype("int16")
    y = y.tolist()

    print("Finished preprocessing")

    # Train, val, test split
    df_train, df_val, df_test = data.split_df(df,
                                              "Patient ID",
                                              split,
                                              total_filter=data_filter)
    df_train = df_train.path.to_list()
    df_val = df_val.path.to_list()
    df_test = df_test.path.to_list()

    print(
        f"Finished reducing and splitting Data. Kept {len(df)*data_filter} records"
    )

    # Make tf.data.Dataset
    ds_train = data.make_dataset(path_to_png, 32, df_train, y)
    ds_val = data.make_dataset(path_to_png, 32, df_val, y)
    ds_test = data.make_dataset(path_to_png, 32, df_test, y, test_set=True)

    classes_dict = pd.DataFrame(mlb.classes_).to_dict()[0]
    classes = mlb.classes_

    print(f"Finished making tf.data.Datasets with classes: {classes}")

    # Trainer()
    model = trainer.Trainer(ds_train, ds_val, job_type)

    # Store trainer split for mlflow params
    model.data_split = data_filter
    model.mlflow_log_param('dataset_filtered', model.data_split)
    model.train_obs = len(df_train)
    model.mlflow_log_param('total_imgs', model.train_obs)
    model.train_val_test = split
    model.mlflow_log_param('train_val_test', model.train_val_test)

    print("Instanciated Trainer()")

    model.build_cnn(
        input_shape=img_size,
        # output_shape=len(classes),
        dense_layer_geometry=(1024, 512, 256),  # Hyperparam
        dropout_layers=dropout_layer,  # Hyperparam
        dropout_rate=dropout_rate, # Hyperparam
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

    print("Evaluating performance")
    results = model.evaluate_model(ds_test, )  #steps=ds_test)

    model.save_model()

    print("Saved model")
