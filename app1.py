from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load models
cnn_model = load_model('trained_model.keras')
with open('decision_tree_model.pkl', 'rb') as model_file:
    decision_tree_model = pickle.load(model_file)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to preprocess the image for the CNN model
def preprocess_image_cnn(image_path):
    image = load_img(image_path, target_size=(128, 128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    return input_arr

# Function to preprocess the image for the Decision Tree model
def preprocess_image_dt(image_path):
    image = load_img(image_path, target_size=(128, 128), color_mode="grayscale")
    input_arr = img_to_array(image)
    input_arr = input_arr.flatten().reshape(1, -1)
    return input_arr

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Normalize the file path for URL usage
            rel_file_path = file_path.replace("\\", "/")
            rel_file_path = rel_file_path.split("static/")[-1]  # Get relative path after 'static/'

            # CNN Prediction
            cnn_input = preprocess_image_cnn(file_path)
            cnn_prediction = cnn_model.predict(cnn_input)
            cnn_result = np.argmax(cnn_prediction)

            # Decision Tree Prediction
            dt_input = preprocess_image_dt(file_path)
            dt_prediction = decision_tree_model.predict(dt_input)
            dt_result = dt_prediction[0]

            # Class names
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                           'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                           'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                           'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                           'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                           'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                           'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

            return render_template('index.html', file_path=rel_file_path, cnn_result=class_names[cnn_result], dt_result=class_names[dt_result])
    
    return render_template('index.html')

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    app.run(debug=True)
