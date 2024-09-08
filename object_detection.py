import tensorflow as tf
import numpy as np
import cv2

# Load pre-trained model
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_320x320/saved_model')

def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb)
    img_tensor = img_tensor[tf.newaxis, ...]

    detections = model(img_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    return boxes, scores

# Example usage
image_path = 'test_image.jpg'
boxes, scores = detect_objects(image_path)
print('Boxes:', boxes)
print('Scores:', scores)
