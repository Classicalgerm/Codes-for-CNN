# Load Model from Github to Colab

!wget --no-check-certificate \
  "https://github.com/Classicalgerm/Codes-for-CNN/blob/main/cifar10_model_adam.h5" \
  -O cifar10_model_adam.h5

from tensorflow.keras.models import load_model

model = load_model("cifar10_model_adam.h5")
print("Model loaded successfully!")

# Test CIFAR-10 accuarcy
_, acc = model.evaluate(x_test, y_test)
print(f"Model Accuracy: {acc:.2f}")

## Presentation

# Upload Image and resize from webots
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import files

uploaded = files.upload()
print("Uploaded files:", list(uploaded.keys()))

import os
print(os.listdir())

# CIFAR-10 labels
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_and_predict(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    print(f"Prediction for {img_path}: {predicted_class}")

load_and_predict("captured_image.png")
