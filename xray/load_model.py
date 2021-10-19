from tensorflow.keras.models import load_model
import os

# model = load_model('/home/santiago/code/Giovita/xray-exam-diagnosis-cnn/models/multilabel/vgg16/2021-10-19_15:08:20.881003')

# model.summary()

for file in os.lsitdir(
        '/home/santiago/code/Giovita/xray-exam-diagnosis-cnn/models/multilabel/vgg16/2021-10-19_16:16:47.144772'
):
    print(file)
