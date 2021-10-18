## agregar columna con el path de cada imagen
import os
from glob import glob


def get_paths(
    dataframe,
    data_folder,
    return_path_col=True,
    return_relative=False,
    verbose = 0,
    overwrite_path=False
    ):

    if 'path' in dataframe.columns and not overwrite_path:
        print("'Path' column already exists")
        return

    all_image_paths = {os.path.basename(x): x for x in glob(
                os.path.join(data_folder, '**/*.png'), recursive=True)}

    if return_path_col:
        dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)
        if return_relative:
            dataframe['path'] = dataframe.path.map(lambda x: "/".join(x.split("/")[-3:]))

    if verbose == 1:
        print('Scans found:', len(all_image_paths), ', Total Headers',
          dataframe.shape[0])
        return all_image_paths

"""
move files from gdrive to gcp using G. Colab

from google.colab import drive
from google.colab import auth

drive.mount('/content/drive')

auth.authenticate_user()
project_id = 'wagon-bootcamp-323816'
!gcloud config set project {project_id}
!gsutil ls

bucket_name = 'xray-lewagon-testupload'
!gsutil -m cp -r /content/drive/My\ Drive/Proyecto_Lewagon_Rayos_X/images_001/* gs://{bucket_name}/images/


"""
