import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load Trained Model
model = tf.keras.models.load_model("soybean_disease_model.h5")

# Load Class Names
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}  # Reverse mapping

IMG_SIZE = (224, 224)

# Function to Predict Image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display Image
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    st.pyplot(plt)

    return predicted_class, confidence

# Streamlit UI
st.title("🌿 Soybean Disease Prediction")
st.write("Upload an image of a soybean leaf to predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    prediction, confidence = predict_image("temp_image.jpg")

    # Display Prediction
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}")
