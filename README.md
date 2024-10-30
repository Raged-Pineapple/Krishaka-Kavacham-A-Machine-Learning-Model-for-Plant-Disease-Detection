Krishaka Kavacham: Plant Disease Detection Model
Krishaka Kavacham is a machine learning project designed to help identify plant diseases quickly and accurately. Utilizing Convolutional Neural Networks (CNN), it classifies uploaded plant images as either "Healthy" or "Diseased" and can identify the specific disease from a set of 38 classes. The project includes a web application interface for easy user interaction.

Table of Contents
Features
Tech Stack
Installation
Usage
Future Scope
License
Features
Plant Disease Classification: The model identifies diseases from 38 classes, including Apple Scab, Grape Black Rot, Corn Leaf Spot, and others.
Severity Estimation: Predicts the level of disease severity, advising users if treatment is needed.
User-Friendly Web Interface: A simple interface for uploading images and receiving diagnostic results in real-time.
Tech Stack
Machine Learning: TensorFlow for CNN model implementation
Backend: Flask for handling HTTP requests and serving the model
Frontend: HTML, CSS, and JavaScript for a responsive web application
Installation
Prerequisites
Python 3.7 or later
pip (Python package manager)
Clone Repository
bash
Copy code
git clone https://github.com/Raged-Pineapple/krishaka-kavacham.git
cd krishaka-kavacham
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Set Up Model and Data
Ensure the trained CNN model and class indices JSON file are located in the models/ directory.
Optionally, add a data/ directory for storing any additional images used for testing or demonstration.
Run the Application
bash
Copy code
python app.py
Access the app locally at http://127.0.0.1:5000.

Usage
Open the web app in your browser.
Upload an image of a plant leaf.
Click "Diagnose Now" to get the disease classification and severity level.
Future Scope
Expand the dataset to include more plant species and disease types.
Make the app mobile-friendly and accessible to a broader audience.
Use data augmentation techniques for improved robustness and accuracy.
License
This project is licensed under the MIT License. See the LICENSE file for details.
