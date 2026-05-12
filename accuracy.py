import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# --- Load Model Architecture + Weights ---
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights("model_weights.h5")

# --- Labels list ---
labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

# --- Test Folder ---
test_dir = "test/"
correct, total = 0, 0

for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)).reshape(1, 48, 48, 1) / 255.0

        pred = model.predict(img, verbose=0)
        pred_label = labels[np.argmax(pred)]

        if pred_label.lower() == folder.lower():
            correct += 1
        total += 1

# --- Print only final accuracy ---
accuracy = (correct / total) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
