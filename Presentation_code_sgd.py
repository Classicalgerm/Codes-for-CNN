from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
from PIL import Image  # For displaying the uploaded image

# 1. Load model from local Colab storage
model = load_model("cifar10_model_sgd.h5")
print("Model loaded successfully!")

# 2. Load CIFAR-10 test data
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = to_categorical(y_test, 10)

# 3. Evaluate model accuracy
_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Model Accuracy: {acc:.2f}")

# 4. CIFAR-10 labels
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 5. Prediction function with visualization
def load_and_predict(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show image + prediction
    plt.imshow(Image.open(img_path))
    plt.axis('off')
    plt.title(f"🧠 Predicted: {predicted_class} ({confidence * 100:.2f}%)")
    plt.show()

# 6. Upload and predict
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
load_and_predict(file_name)
