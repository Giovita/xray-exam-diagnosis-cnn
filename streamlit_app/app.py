import streamlit as st
import tensorflow as tf
import cv2
import time
# from PIL import Image, ImageOps
import numpy as np
from PIL import Image




#-------------------------------
#    Load Models functions
#-------------------------------

#@st.cache(allow_output_mutation=True)
def load_model_binario():
    model = tf.keras.models.load_model('streamlit_app/models_binary_vgg16_2021-10-20_03_44_11.263701')
    return model


def load_model_multiple():
    model = tf.keras.models.load_model(
        'streamlit_app/models_multilabel_vgg16_2021-10-20_03_47_38.813584')
    return model



#-------------------------------
#    Input imagen del usuario
#-------------------------------
st.markdown("""
    # A Chest X-ray Diagnosis ⚕️
    A Deep Neural Network trained to detect fourteen common thoracic pathologies by analyzing a patient Chest X-ray image.

    -------------------------------------------------------------------------------------------------------------------"""
            )
st.markdown('##### 📤   Please Upload a Chest X-ray Image')
file = st.file_uploader("jpg/png image", type=["jpg", "png"])
#st.set_option('deprecation.showfileUploaderEncoding', False)

#-------------------------------
#   Resize image, Cargar modelo y Predecir
#-------------------------------

img_size = (224, 224)

if file is None:
    pass
else:
    if file is not None:

        # Carga del primer modelo (BINARIO)
        with st.spinner('Finding some disease... ⌛ '):
            time.sleep(1)
            model_binario = load_model_binario()

        #st.text("")
        st.markdown(""" -------------------------------------------------------------------------------------------------------------------""")

        st.write('## Patient X-ray Image')

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, width=550, channels="BGR")

        img = cv2.resize(opencv_image,(224, 224))  # resize image to match model's expected sizing
        img = np.reshape(img,[1, 224, 224, 3])  # return the image with shaping that TF wants.

        prediction = model_binario.predict(img)
        if prediction == 0:
            st.markdown('### 🩺  The patient is probably not sick ')
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

            st.markdown('### 🩺  The patient could be ill with:')
            # Carga del primer modelo (MULTIPLE)
            with st.spinner(
                    'Detecting thoracic pathologies... ⌛'):

                time.sleep(1)
                model_multiple = load_model_multiple()
            prediction2 = model_multiple.predict(img)

            #st.write(prediction2[0] >= 0.5)
            for idx, val in enumerate(prediction2[0]):
                if val > 0.5:
                    disease = multi_class[idx]
                    st.markdown(
                        f"#### 🦠 {disease.upper()} thoracic phatology")
                else:
                    st.markdown(
                        """#### ❌ Sorry, It was imposible to determine a thoracic phatology
                We suggest you to do more tests on the patient 🧪    """)
                    break

            #line space
            for x in range(3):
                st.text("")

            st.markdown(""" -------------------------------------------------------------------------------------------------------------------""")

            st.markdown('##### 🔄 Refresh page for a new pathology detection')

            #line space
            for x in range(2):
                st.text("")

            st.caption(
                'This tool was developed by students at Lewagon Datascience Bootcampp'
            )

            image = Image.open('streamlit_app/wagon.png')
            st.image(image, caption='Le Wagon', use_column_width=False)









#st.experimental_rerun()
#st.write(prediction2)
#with {val} certainty
