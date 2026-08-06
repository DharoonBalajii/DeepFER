"""Predict the emotion in a single face image.

Usage:
    python predict_image.py path/to/face.jpg
"""

import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import EMOTIONS, IMG_SIZE

MODEL_PATH = "outputs/deepfer_model.keras"


def predict(image_path):
    model = load_model(MODEL_PATH)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = model.predict(img)[0]
    top = np.argmax(probs)

    print(f"\nPredicted emotion: {EMOTIONS[top]} ({probs[top] * 100:.1f}% confidence)\n")
    for emotion, p in sorted(zip(EMOTIONS, probs), key=lambda x: -x[1]):
        print(f"  {emotion:10s} {p * 100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py path/to/face.jpg")
        sys.exit(1)
    predict(sys.argv[1])
