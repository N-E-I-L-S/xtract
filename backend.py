# import os
# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load your pre-trained CNN model
# model = load_model('./cnn_lung_cancer.h5')

# # Preprocess function to resize and normalize the image
# def preprocess_image(image):
#     # Resize the image to the expected input size
#     image = cv2.resize(image, (80, 80))

#     # Normalize the image
#     image = image.astype('float32') / 255.0

#     # Add an extra dimension for the batch size
#     image = np.expand_dims(image, axis=0)

#     return image

# # API endpoint to receive an image and make predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the image file from the request
#     file = request.files['image']

#     # Read the image file
#     image_bytes = file.read()
#     image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

#     # Preprocess the image
#     preprocessed_image = preprocess_image(image)

#     # Make predictions
#     predictions = model.predict(preprocessed_image)

#     # Get the class with the highest probability
#     predicted_class = np.argmax(predictions[0])

#     # Map the predicted class to a label
#     labels = ['benign', 'adenocarcinoma', 'squamous_cell_carcinoma']
#     predicted_label = labels[predicted_class]

#     # Return the prediction as a JSON response
#     response = {'prediction': predicted_label}
#     return jsonify(response)


# if __name__ == '__main__':
#     app.run()

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

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

# Function to make predictions
def predict_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions[0])

    # Map the predicted class to a label
    labels = ['benign', 'adenocarcinoma', 'squamous_cell_carcinoma']
    predicted_label = labels[predicted_class]

    return predicted_label

# Streamlit app
def main():
    st.title('Lung Cancer Prediction')
    st.write('Upload an image and get the predicted result.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Read the image file
        image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        prediction = predict_image(image)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
