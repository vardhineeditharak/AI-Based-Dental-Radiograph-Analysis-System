from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 160
CLASSES = ["Cavity", "Fillings", "Impacted", "Implant", "Normal"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "dental_model.h5"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"TensorFlow model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image uploaded")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return CLASSES[idx], confidence
