import streamlit as st
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_URL = "https://drive.google.com/file/d/10j5UikpbPJfaNRcW04_eBdTeuWum4Ozw/view?usp=sharing"
MODEL_PATH = "weather_model.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = download_and_load_model()
