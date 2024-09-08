import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained model
model = load_model('hand_gesture_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return np.argmax(prediction)

# Example usage
image_path = 'test_hand_gesture.jpg'
gesture = predict_gesture(image_path)
print(f'Predicted gesture: {gesture}')
