# -----------------
# GCP Params
# -----------------

PROJECT_ID = 'wagon-bootcamp-323816'

BUCKET_NAME = 'xray-lewagon-testupload'

REGION = 'SOUTHAMERICA-EAST1'

BUCKET_TRAIN_DATA_PATH = 'images/'

BUCKET_TRAIN_CSV_PATH = ''

GCP_IMAGE_BUCKET = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"

GCP_MODEL_STORAGE_LOCATION = 'models/'

GCP_MODEL_BUCKET = f'gs://{BUCKET_NAME}/{GCP_MODEL_STORAGE_LOCATION}'

# -----------------
# Predict Params
# -----------------

AWS_BUCKET_TEST_PATH = None

# -----------------
# MLFlow
# -----------------

MLFLOW_URI = "https://mlflow.lewagon.co/"

EXPERIMENT_NAME = "[AR] [BS AS] [xray-diagnosis]"

# -----------------
# Local_storage
# -----------------

PATH_TO_LOCAL_MODEL = 'models/'

# -----------------
# Model Variables
# -----------------

"""DEPRECATED. Not to be used. """

MODEL_NAME = 'xray-diagnosis'

MODEL_VERSION = 'v1'

PATH_TO_GCP_MODEL = f'gs://{BUCKET_NAME}/models/{MODEL_NAME}/{PATH_TO_LOCAL_MODEL}'
