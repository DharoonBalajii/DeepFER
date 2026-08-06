# DeepFER: Facial Emotion Recognition Using Deep Learning

A beginner-friendly CNN project that classifies a face image into one of
seven emotions: **angry, disgust, fear, happy, neutral, sad, surprise**.

Built to follow the DeepFER project brief: data collection & augmentation,
a CNN trained from scratch, evaluation with accuracy/precision/recall/F1,
and a real-time webcam demo.

## Project structure

```
DeepFER/
├── dataset/            # train/ and test/ folders, one subfolder per emotion
├── model.py             # CNN architecture
├── train.py              # trains the model, saves it + accuracy/loss plots
├── evaluate.py            # precision/recall/F1 + confusion matrix on test set
├── predict_image.py        # predict the emotion in a single photo
├── realtime.py             # live webcam emotion detection
├── outputs/                # trained model, plots, metrics (created after training)
└── requirements.txt
```

## 1. Setup

```bash
cd DeepFER
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train.py --epochs 25
```

This loads images from `dataset/train`, applies data augmentation
(random rotation, zoom, shifts, and horizontal flips) to help the model
generalize, and trains the CNN in `model.py`. Since `disgust` has far
fewer training images than the other classes, the training script
weights rarer classes more heavily so the model doesn't ignore them.

The best model (by validation accuracy) is saved to
`outputs/deepfer_model.keras`, and `outputs/training_history.png` shows
accuracy/loss curves per epoch.

## 3. Evaluate on the test set

```bash
python evaluate.py
```

Prints per-class precision, recall, and F1-score, and saves
`outputs/confusion_matrix.png`.

## 4. Try it on a single photo

```bash
python predict_image.py path/to/a/face.jpg
```

## 5. Real-time webcam demo

```bash
python realtime.py
```

Opens your webcam, detects faces with OpenCV's Haar cascade, and labels
each face with its predicted emotion live. Press `q` to quit.

## About the dataset

7 emotion classes, ~36k grayscale 48x48 face crops (28,709 for training,
7,178 for testing), collected from public facial-expression datasets.
Class sizes are imbalanced (e.g. many more "happy" images than
"disgust"), which is why training uses class weighting.

## Notes for going further

- The CNN here is trained from scratch. The project brief also mentions
  transfer learning (fine-tuning a pre-trained model like MobileNetV2) as
  a way to boost accuracy further — a natural next step once this
  baseline is working.
- `outputs/` is gitignored since trained models and plots are generated
  artifacts, not source code.
