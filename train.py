"""Train the DeepFER emotion recognition CNN.

Usage:
    python train.py [--epochs 25] [--batch-size 64]

Loads images from dataset/train and dataset/test, applies data
augmentation (rotation, zoom, shifts, horizontal flip) to the training
set, trains the CNN defined in model.py, and saves:
    outputs/deepfer_model.keras   - the trained model
    outputs/training_history.png  - accuracy/loss curves
    outputs/history.json          - raw per-epoch metrics
"""

import argparse
import json
import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from model import EMOTIONS, IMG_SIZE, build_model

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def get_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=True,
    )
    val_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        classes=EMOTIONS,
        shuffle=False,
    )
    return train_gen, val_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_gen, val_gen = get_generators(args.batch_size)

    # The dataset is imbalanced (e.g. "disgust" has far fewer images than
    # "happy"), so weight rarer classes higher during training.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "deepfer_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    plot_history(history.history)
    print(f"\nTraining complete. Best model saved to {OUTPUT_DIR}/deepfer_model.keras")


def plot_history(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)


if __name__ == "__main__":
    main()
