"""Real-time facial emotion recognition from a webcam.

Usage:
    python realtime.py

Uses OpenCV's Haar cascade to find faces in each webcam frame, then
runs the trained DeepFER CNN on each face crop. Press 'q' to quit.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import EMOTIONS, IMG_SIZE

MODEL_PATH = "outputs/deepfer_model.keras"

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def main():
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face.astype("float32") / 255.0
            face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            probs = model.predict(face, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(probs)]
            confidence = np.max(probs) * 100

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            label = f"{emotion} ({confidence:.0f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 0), 2)

        cv2.imshow("DeepFER - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
