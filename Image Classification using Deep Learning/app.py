from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')  # Make sure to save your trained model as 'model.h5'

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_path = './temp.jpg'
    file.save(img_path)
    
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return jsonify({'class': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
