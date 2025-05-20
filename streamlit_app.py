import streamlit as st
import numpy as np
import os
import urllib.request
import sys
from PIL import Image
import onnxruntime as ort

# -- Configuration --
st.set_page_config(page_title="Weather Classifier 🌤️", layout="centered")
st.title("🌦️ Weather Image Classifier (ONNX)")
st.write("Upload an image to classify it as one of the weather conditions using a MobileNetV2 ONNX model.")

st.write("Python version:", sys.version)

# URL to ONNX model (must be a direct download link, not Google Drive)
MODEL_URL = "https://your-direct-link/weather_classification.onnx"
MODEL_PATH = "weather_classification.onnx"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Cloudy', 'Rainy', 'Shine', 'Sunrise']  # Make sure these match training labels

# -- Download and load ONNX model --
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session

session = download_and_load_model()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -- Upload and predict --
uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Classifying..."):
        preds = session.run([output_name], {input_name: img_array})[0]
        predicted_class = np.argmax(preds[0])
        confidence = preds[0][predicted_class] * 100
        st.success(f"**Prediction:** {CLASS_NAMES[predicted_class]} ({confidence:.2f}% confidence)")
