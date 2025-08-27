import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# --------------------------
# Load Model in Cache
# --------------------------
# Google Drive direct download link
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1Aa9YKA1cwWRVN1peaKEWsdcnXW5hxXUU"
OUTPUT_PATH = "model.h5"

def download_file_from_google_drive(url, output_path):
    """Download file from Google Drive direct link."""
    if not os.path.exists(output_path):
        st.info("Downloading model, please wait...")
        session = requests.Session()
        response = session.get(url, stream=True)
        token = None

        # Check for confirmation token for large files
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value

        if token:
            url_with_token = url + "&confirm=" + token
            response = session.get(url_with_token, stream=True)

        # Save file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully!")

@st.cache_resource
def load_model():
    # Ensure the model is downloaded
    download_file_from_google_drive(DOWNLOAD_URL, OUTPUT_PATH)

    # Load the TensorFlow model
    model = tf.keras.models.load_model(OUTPUT_PATH)
    return model

model = load_model()

# --------------------------
# Class Labels and Threshold
# --------------------------
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
CONFIDENCE_THRESHOLD = 0.7

# --------------------------
# Helper Functions
# --------------------------
def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img, verbose=0)
    confidence = float(np.max(preds))
    label = class_names[int(np.argmax(preds))]
    if confidence < CONFIDENCE_THRESHOLD or label == "nothing":
        return None, confidence
    return label, confidence

# --------------------------
# Streamlit UI
# --------------------------
st.title("🤟 ASL Hand Sign Recognition")
st.write("Upload an image or use your webcam to predict ASL hand signs.")

option = st.radio("Choose Input Method:", ("Upload Image", "Use Webcam"))

# Upload Image Option
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a Hand Sign Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, confidence = predict(image)
        if label:
            st.success(f"Predicted Sign: **{label}** (Confidence: {confidence:.2f})")
        else:
            st.warning("⚠️ No valid hand sign detected.")

# Webcam Option
elif option == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture).convert("RGB")
        st.image(img, caption="Captured Image", use_column_width=True)

        label, confidence = predict(img)
        if label:
            st.success(f"Predicted Sign: **{label}** (Confidence: {confidence:.2f})")
        else:
            st.warning("⚠️ No valid hand sign detected.")
