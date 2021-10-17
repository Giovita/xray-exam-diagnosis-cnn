## agregar columna con el path de cada imagen
import os
from glob import glob


def get_paths(
    dataframe,
    data_folder,
    return_path_col=True,
    return_relative=False,
    verbose = 0,

):
    all_image_paths = {os.path.basename(x): x for x in glob(
                os.path.join(data_folder, '**/*.png'), recursive=True)}

    if return_path_col:
        dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)
        if return_relative:
            dataframe.path.map(lambda x: "/".join(x.split("/")[-3:]))

    if verbose == 1:
        print('Scans found:', len(all_image_paths), ', Total Headers',
          dataframe.shape[0])
        return all_image_paths
