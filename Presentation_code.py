# Download the trained model from GitHub (RAW file, use ?raw=true)
!wget --no-check-certificate \
  "https://github.com/Classicalgerm/Codes-for-CNN/raw/main/cifar10_model.h5" \
  -O cifar10_model.h5

# Load the model
from tensorflow.keras.models import load_model
model = load_model("cifar10_model.h5")
print("Model loaded successfully!")

# Load CIFAR-10 test data to evaluate model performance
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = y_test.flatten()

# Evaluate model accuracy
_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Model Accuracy on CIFAR-10 Test Set: {acc:.2f}")

# Upload and predict on a Webots image
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import files
import os

# Upload Webots-captured image
uploaded = files.upload()
print("Uploaded files:", list(uploaded.keys()))
print("Current directory contents:", os.listdir())

# CIFAR-10 class labels
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Prediction function
def load_and_predict(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    print(f"Prediction for {img_path}: {predicted_class}")

# Call prediction function for uploaded image
load_and_predict("captured_image.png")
