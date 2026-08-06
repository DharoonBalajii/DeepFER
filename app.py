"""DeepFER web app.

Upload a photo or take one with your camera, and the trained CNN
predicts the emotion. Run locally with:

    streamlit run app.py

or deploy it on Streamlit Community Cloud pointed at this repo.
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from model import EMOTIONS, IMG_SIZE

MODEL_PATH = "outputs/deepfer_model.keras"

EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def get_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict_emotion(image: Image.Image):
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = get_face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        face = gray
        face_found = False
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
        face_found = True

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = get_model().predict(face, verbose=0)[0]
    return probs, face_found


st.set_page_config(page_title="DeepFER", page_icon="🙂")

st.title("DeepFER: Facial Emotion Recognition")
st.write(
    "Upload a photo or take one with your camera. A CNN trained on ~36k "
    "labeled faces predicts one of 7 emotions."
)

source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)
if source == "Upload a photo":
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo")

if image_file is not None:
    image = Image.open(image_file)
    probs, face_found = predict_emotion(image)

    if not face_found:
        st.warning("No face detected — predicting on the whole image instead.")

    top = int(np.argmax(probs))
    st.image(image, caption="Input image", width=300)
    st.subheader(f"{EMOJIS[EMOTIONS[top]]} {EMOTIONS[top].capitalize()} — {probs[top] * 100:.1f}% confidence")

    chart_data = pd.DataFrame({"probability": probs}, index=[e.capitalize() for e in EMOTIONS])
    st.bar_chart(chart_data)
