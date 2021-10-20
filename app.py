import streamlit as st
import tensorflow as tf
import cv2
import time
# from PIL import Image, ImageOps
import numpy as np
# from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import preprocess_input


# Load Models functions
#@st.cache(allow_output_mutation=True)
def load_model_binario():
    model = tf.keras.models.load_model('streamlit/my_model.hdf5')
    return model


def load_model_multiple():
    model = tf.keras.models.load_model(
        'streamlit/vgg16/2021-10-19_15_08_20.881003')
    return model


# Prompt de la imagen al usuario
st.subheader('Upload a chest X-ray image')
file = st.file_uploader("jpg or png image", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
# Size de la imagen para el modelo
img_size = (224, 224)
# Codigo a ejecutar :  Resize image, Cargar modelo y Predecir
if file is None:
    st.text("Please upload a chest X-ray image")
else:
    if file is not None:
        # Carga del primer modelo (BINARIO)
        with st.spinner(
                'Se esta cargando el modelo que predice si esta enfermo o no lo esta...'
        ):
            time.sleep(1)
            model_binario = load_model_binario()
        st.write("""# Chest X-ray Classification""")
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
        img = cv2.resize(
            opencv_image,
            (224, 224))  # resize image to match model's expected sizing
        img = np.reshape(
            img,
            [1, 224, 224, 3])  # return the image with shaping that TF wants.
        prediction = model_binario.predict(img)
        if prediction == 0:
            binary_class = ['No_Enfermo', 'Enfermo']
            st.write(binary_class[int(round(prediction[0][0]))])
            st.write(int(round(prediction[0][0])))
            st.subheader('Paciente No Enfermo')
        else:
            multi_class = {
                0: 'Atelectasis',
                1: 'Cardiomegaly',
                2: 'Consolidation',
                3: 'Edema',
                4: 'Effusion',
                5: 'Emphysema',
                6: 'Fibrosis',
                7: 'Hernia',
                8: 'Infiltration',
                9: 'Mass',
                10: 'Nodule',
                11: 'Pleural_Thickening',
                12: 'Pneumonia',
                13: 'Pneumothorax'
            }
            st.write('El Paciente Esta Enfermo 😞')
            # Carga del primer modelo (MULTIPLE)
            with st.spinner(
                    'Se esta cargando el modelo que predice que enfermedad tiene...'
            ):
                time.sleep(1)
                model_multiple = load_model_multiple()
            prediction2 = model_multiple.predict(img)
            st.write(prediction2)
            #st.write(prediction2[0] >= 0.5)
            for idx, val in enumerate(prediction2[0]):
                if val > 0.5:
                    disease = multi_class[idx]
                    st.write(f"Patient has {disease} with {val} certainety")
