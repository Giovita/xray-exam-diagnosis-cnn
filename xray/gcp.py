import os

from google.cloud import storage
from xray.params import BUCKET_NAME, GCP_STORAGE_LOCATION


def storage_upload(
    model_name,
    base_model,
    version,
    rm=False,
):
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = f"{model_name}"
    storage_location = f"models/{base_model}/{version}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(local_model_name)
    print(
        f"=> {local_model_name} uploaded to bucket {BUCKET_NAME} inside {storage_location}",
        "green",
    )
    if rm:
        os.remove(local_model_name)
