<<<<<<< HEAD
=======
"""
EDIT FOR USE IN XRAY PROJECT! defaults from challenge in place.
"""

>>>>>>> 51d9592 (Refactor Data Pipeline, move  to data.py)
### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[AR] [BS AS] [Giovita] taxifaremodel 02"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-710-giovagnoli'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
# BUCKET_TRAIN_DATA_PATH_FULL = ' '  # Fill with path to full 5 GB dataset

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
