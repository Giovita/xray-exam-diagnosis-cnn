import os
import pickle
from urllib.parse import urljoin

import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from werkzeug import Request

# Load app auth credentials from .env.
load_dotenv()

# Setup Client
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


def load_model_binary_classification():
    model = tf.keras.models.load_model("models/models_binary_vgg16_2021-10-20_03_44_11.263701")
    return model


def load_model_multilabel():
    model = tf.keras.models.load_model("models/models_multilabel_vgg16_2021-10-20_03_47_38.813584")
    return model


def decode_image(file):
    """
    Decodes recieved binary into original format and process it for TF model use
    """
    if not file:
        print("NO QUIERE")
    if file:
        # img = pickle.loads(file)
        # return file.read()
        print("va queriendo")
        file_bytes = np.asarray(bytearray(file), dtype=np.uint8)
        print(file_bytes)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(opencv_image, (224, 224))  # resize image to match model's expected sizing
        img = np.reshape(img, [1, 224, 224, 3])
        return img


def classify_image(img):
    """
    Runs the passed image through the prediction models, returning a json object
    with the result as response
    """
    # return img_to_process
    binary_model = load_model_binary_classification()
    binary_prediction = binary_model.predict(img)

    if binary_prediction == 0:
        response = {"diagnostic": None}
        return response

    multi_class = {
        0: "Atelectasis",
        1: "Cardiomegaly",
        2: "Consolidation",
        3: "Edema",
        4: "Effusion",
        5: "Emphysema",
        6: "Fibrosis",
        7: "Hernia",
        8: "Infiltration",
        9: "Mass",
        10: "Nodule",
        11: "Pleural_Thickening",
        12: "Pneumonia",
        13: "Pneumothorax",
    }
    multi_model = load_model_multilabel()
    multi_prediction = multi_model.predict(img)

    diseases = [f"{multi_class[idx]}" for idx, val in enumerate(multi_prediction[0]) if val > 0.5]
    if len(diseases) > 0:
        return {"diagnostic": " ".join(diseases)}

    return {
        "diagnostic": """❌ Sorry, It was imposible to determine a thoracic phatology.
                        We suggest you to do more tests on the patient 🧪"""
    }


@app.post("/api/v1/make-prediction", status_code=200)
async def make_prediction(file: UploadFile = File(...)):
    # async def make_prediction(file: bytes = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    img = decode_image(await file.read())
    # prediction = classify_image(img)
    # return prediction
    return {
        "diagnostic": """❌ Sorry, It was imposible to determine a thoracic phatology.
                        We suggest you to do more tests on the patient 🧪"""
    }
