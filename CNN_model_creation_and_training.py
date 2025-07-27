# CNN model creation

# Step 1: Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 2: Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Step 3: Normalize the images to 0-1 range
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 4: Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 5: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes in CIFAR-10
])

# Step 6: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
--------------------------------------------------------
# Step 8: Check the Accuracy After Training
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Show model architecture
model.summary()
# Evaluate test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
-------------------------------------------------------
#Step 9: Test with images 
 ##- RGB images 32x32 pixel ; Upload the Images to Colab on the left side of Colab, click the folder icon → then click the Upload button and upload 3 images (e.g., cat1.jpg, car1.jpg, ship1.jpg).
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

# Test 3 custom images; Screenshot each prediction and save it for report/presentation.
#load_and_predict("airplane1.jpg")
#load_and_predict("automobile1.jpg")
#load_and_predict("cat1.jpg")
load_and_predict("captured_image.png")
-------------------------------------------------------
# Step 10: Save the model
model.save("cifar10_model.h5")
 ## download to PC
from google.colab import files
files.download("cifar10_model.h5")
