from json import load
from tensorflow.keras.models import load_model
import os
from google.cloud import storage
from tensorflow.python.lib.io.file_io import file_crc32
from xray.params import (
    GCP_MODEL_STORAGE_LOCATION,
    BUCKET_NAME,
    MLFLOW_URI,
    EXPERIMENT_NAME,
    BASE_MODEL_FOLDER,
    PATH_TO_LOCAL_MODEL,
    CHECKPOINT_FOLDER,
)
from pathlib import Path

# model = load_model('/home/santiago/code/Giovita/xray-exam-diagnosis-cnn/models/multilabel/vgg16/2021-10-19_15:08:20.881003')

# model.summary()

# for file in os.lsitdir(
#         '/home/santiago/code/Giovita/xray-exam-diagnosis-cnn/models/multilabel/vgg16/2021-10-19_16:16:47.144772'
# ):
#     print(file)


def load_model_from_gcp(origin_model_dir, filename, dest_dir):
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

    blob.download_to_filename(f"{filepath}.h5")

    # if os.path.isfile(local_path):
    return load_model(f"{filepath}.h5")


if __name__ == "__main__":
    model_dir = "models/binary/vgg16/"
    filename = "2021-10-20_03:44:11.263701"
    dest_dir = os.path.join(os.getcwd(), model_dir)

    model = load_model_from_gcp(model_dir, filename, dest_dir)
    model.summary()
