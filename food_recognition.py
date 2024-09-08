import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

def recognize_food(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Example usage
image_path = 'test_food.jpg'
predictions = recognize_food(image_path)
for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f'{i+1}: {label} ({score:.2f})')
