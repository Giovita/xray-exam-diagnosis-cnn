import pickle
import time
from io import StringIO

import cv2

# from PIL import Image, ImageOps
import numpy as np
import requests
import streamlit as st

# -------------------------------
st.markdown(
    """
    # A Chest X-ray Diagnosis ⚕️
    A Deep Neural Network trained to detect fourteen common thoracic pathologies by analyzing a patient Chest X-ray image.

    -------------------------------------------------------------------------------------------------------------------"""
)
st.markdown("##### 📤   Please Upload a Chest X-ray Image")
file = st.file_uploader("jpg/png image", type=["jpg", "png"])
# st.set_option('deprecation.showfileUploaderEncoding', False)

# -------------------------------
#   Resize image, Cargar modelo y Predecir
# -------------------------------

img_size = (224, 224)

if file:
    # st.write(file.getvalue())
    # Carga del primer modelo (BINARIO)
    # file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    # opencv_image = cv2.imdecode(file_bytes, 1)
    # img = cv2.resize(opencv_image, (224, 224))  # resize image to match model's expected sizing
    # img = np.reshape(img, [1, 224, 224, 3])
    # st.markdown(file)
    # img_binary = pickle.dumps(file)
    with st.spinner("Finding some disease... ⌛ "):
        time.sleep(1)

        # return the image with shaping that TF wants.

        url = "http://localhost:8000/api/v1/make-prediction"

        r = requests.post(url, files={"file": file})
        result = r.json()
        print(result)

        if len(result) > 0 and not result["diagnostic"]:
            st.markdown("### 🩺  The patient is probably not sick ")
        else:
            st.markdown(f"The patient could be ill with {result['diagnostic']}")
