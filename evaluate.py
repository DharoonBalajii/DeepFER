"""Evaluate the trained DeepFER model on the held-out test set.

Usage:
    python evaluate.py

Reports accuracy, per-class precision/recall/F1, and saves a confusion
matrix image to outputs/confusion_matrix.png.
"""

import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import EMOTIONS, IMG_SIZE

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "deepfer_model.keras")


def main():
    model = load_model(MODEL_PATH)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False,
    )

    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)
    print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")


def plot_confusion_matrix(cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("DeepFER Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)


if __name__ == "__main__":
    main()
