import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained CNN model
model = load_model('./cnn_lung_cancer.h5')

# Preprocess function to resize and normalize the image
def preprocess_image(image):
    # Resize the image to the expected input size
    image = cv2.resize(image, (80, 80))

    # Normalize the image
    image = image.astype('float32') / 255.0

    # Add an extra dimension for the batch size
    image = np.expand_dims(image, axis=0)

    return image

# API endpoint to receive an image and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Read the image file
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions[0])

    # Map the predicted class to a label
    labels = ['benign', 'adenocarcinoma', 'squamous_cell_carcinoma']
    predicted_label = labels[predicted_class]

    # Return the prediction as a JSON response
    response = {'prediction': predicted_label}
    return jsonify(response)

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('/f.html')

if __name__ == '__main__':
    app.run()