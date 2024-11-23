from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model#type:ignore
import numpy as np
import cv2
import json
from PIL import Image
import os
from random import randint

app = Flask(_name_)

# Global variables
model = None
class_indices = None
Alpha = None

# Serve the HTML file
@app.route('/')
def serve_html():
    return send_from_directory('', 'el_vidyuth.html')

# Load the saved model
@app.route('/load_model', methods=['POST'])
def load_saved_model():
    global model
    try:
        model_path = r'C:\RVCE(2023-27)\venv\Plantdisease-1.h5'
        model = load_model(model_path)
        return jsonify({"message": "Model loaded successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load class indices

@app.route('/load_class_indices', methods=['POST'])
def load_class_indices():
    global class_indices
    try:
        class_indices_path = r'C:\RVCE(2023-27)\venv\class_indices (1).json'
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        return jsonify({"message": "Class indices loaded successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_alpha(image):
    alpha = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 0] > 200 and image[i, j, 1] > 200 and image[i, j, 2] > 200:
                alpha[i, j] = 255
            else:
                alpha[i, j] = 0
    return alpha

def display_disease_percentage(disease, alpha, threshold):
    count = 0
    res = 0
    for i in range(disease.shape[0]):
        for j in range(disease.shape[1]):
            if alpha[i, j] == 0:
                res += 1
            if disease[i, j] < threshold:
                count += 1
    percent = (count / res) * 100 if res != 0 else 0
    return round(percent, 2)

# Process the selected image
@app.route('/process_image', methods=['POST'])
def process_image():
    global model, class_indices, Alpha
    if model is None:
        return jsonify({"error": "Model not loaded. Please load the model first."}), 400

    if class_indices is None:
        return jsonify({"error": "Class indices not loaded. Please load the class indices first."}), 400

    try:
        # Get the file from the request
        file = request.files['image']
        img = Image.open(file)
        img = img.convert('RGB')
        img = np.array(img)

        # Extract channels
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        Disease = r - g

        # Calculate Alpha channel
        Alpha = get_alpha(img)
        

        # Process your image here with the loaded model
        img_resized = cv2.resize(img, (224, 224))  # Resize as per your model's input shape

        # Preprocess the image
        img_preprocessed = img_resized.astype('float32') / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        # Example prediction
        prediction = model.predict(img_preprocessed)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_indices[str(predicted_class_index)]

        # Determine percentage disease based on class name
        if 'healthy' in predicted_class_name.lower():
            percentage_disease = randint(0, 5)
            message = "The plant is healthy."
        else:
            message = "The plant needs to be treated."
            # Calculate the disease percentage
            threshold = 150  # Default threshold value
            percentage_disease = display_disease_percentage(Disease, Alpha, threshold)

        return jsonify({
            "class": predicted_class_name,
            "percentage_disease": f"{percentage_disease}",
            "message": message
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if _name_ == '_main_':
    app.run(debug=True)
