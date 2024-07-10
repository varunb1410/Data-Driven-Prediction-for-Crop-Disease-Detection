import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('trained_model.keras')

# Load the class names
validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)
class_names = validation_set.class_names

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    return np.array([input_arr])  # Convert single image to a batch

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        # Get the image file from the request
        image_file = request.files['image']
        # Save the file to a temporary location
        image_path = os.path.join('temp', image_file.filename)
        image_file.save(image_path)
    elif 'image_path' in request.form:
        # Get the image path from the request
        image_path = request.form['image_path']
    else:
        return jsonify({"error": "No image or image path provided"}), 400

    # Preprocess the image
    input_arr = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    model_prediction = class_names[result_index]

    return jsonify({"prediction": model_prediction})

if __name__ == '__main__':
    # Create the temp directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(host='0.0.0.0', port=5000)
