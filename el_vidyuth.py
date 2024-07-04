from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
from flask_cors import CORS
from random import randint

# Download the model file if not already present
model_path = 'Plantdisease-1.h5'
if not os.path.exists(model_path):
    url = 'https://github.com/AdityaAdi07/EL_Vidyuth/raw/main/Plantdisease-1.h5'
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_path, 'wb') as file:
            file.write(response.content)
    else:
        print('Failed to retrieve the file:', response.status_code)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model globally
model = None  # Initially set to None until loaded
class_indices = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry_(including_sour)___Powdery_mildew",
    "6": "Cherry_(including_sour)___healthy",
    "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize)___Common_rust_",
    "9": "Corn_(maize)___Northern_Leaf_Blight",
    "10": "Corn_(maize)___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca_(Black_Measles)",
    "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing_(Citrus_greening)",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper,_bell___Bacterial_spot",
    "19": "Pepper,_bell___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites Two-spotted_spider_mite",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato___Tomato_mosaic_virus",
    "37": "Tomato___healthy"
}

@app.before_first_request
def load_model_and_class_indices():
    global model
    try:
        # Load the model
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Serve the HTML file
@app.route('/')
def serve_html():
    return send_from_directory('', 'index.html')

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
    global model, class_indices
    if model is None:
        return jsonify({"error": "Model not loaded. Please load the model first."}), 400

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

if __name__ == '__main__':
    app.run(debug=True)
