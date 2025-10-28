import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# --------- Configuration ---------
MODEL_DIR = 'models/'
MODEL_FILES = [
    'Koala_01_20.h5',
    'Koala_0005_40.h5',
    'Koala_0005_3.h5',
    'Koala_0005_100.h5',
    'Koala_0005_10.h5',
    'Koala_0001_50.h5',
    'Koala_00001_20.h5'
]
# Load models at startup for performance
@st.cache(allow_output_mutation=True)
def load_models():
    models = {}
    for model_file in MODEL_FILES:
        model_path = os.path.join(MODEL_DIR, model_file)
        models[model_file] = load_model(model_path)
    return models

models = load_models()

st.set_page_config(page_title="Manga Colorization Demo", layout="wide")

st.title("Manga Colorization using Deep Koalarization")
st.write("Upload a grayscale manga image, select a trained model, and see it colorized in real-time.")

# Select model
model_choice = st.selectbox("Choose a trained model:", MODEL_FILES)

# Upload image
uploaded_image = st.file_uploader("Upload your grayscale manga image:", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_image is not None and model_choice:
    # Read image
    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if original_img is None:
        st.error("Could not read the image. Please upload a valid image file.")
    else:
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        
        # Resize to 299x299 and preprocess (original method)
        img_resized = cv2.resize(original_img, (299, 299))
        img_inception = preprocess_input(img_resized.copy().astype(np.float32))
        
        # Extract Lab components
        img_lab = cv2.cvtColor(img_inception, cv2.COLOR_RGB2Lab)
        L_channel = img_lab[:, :, 0]  # L channel
        ab_channels = img_lab[:, :, 1:]  # a and b channels

        # Prepare inputs for model
        # Input 1: L channel (normalized to [-1,1] for Inception)
        X_inception = np.repeat(L_channel[:, :, np.newaxis], 3, axis=2)
        X_inception = preprocess_input(X_inception)

        # Input 2: the original resized LAB for encoder
        encoder_input = cv2.resize(img_lab, (224, 224))[:, :, 0]
        encoder_input = encoder_input / 255.0  # normalize to [0,1]

        # Expand dims for batch
        X_inception = np.expand_dims(X_inception, axis=0)
        encoder_input = np.expand_dims(encoder_input, axis=0)
        encoder_input = np.expand_dims(encoder_input, axis=3)  # Add channel dim

        # Load selected model
        selected_model = models[model_choice]

        # Perform inference
        # Assuming model accepts [X_inception, encoder_input]
        try:
            ab_output = selected_model.predict([X_inception, encoder_input])[0]
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            st.stop()

        # Post-process output
        # ab_output is scaled between -1 and 1
        # Resize back to original image size
        ab_resized = cv2.resize(ab_output, (original_img.shape[1], original_img.shape[0]))
        L_value = cv2.cvtColor(original_img, cv2.COLOR_BGR2Lab)[:, :, 0]
        L_normalized = (L_value / 255.0) * 100  # Convert to [0,100] LAB scale

        # Prepare LAB image
        lab_color = np.zeros((original_img.shape[0], original_img.shape[1], 3))
        lab_color[:, :, 0] = L_value  # Original L
        lab_color[:, :, 1:] = ab_resized * 128  # Assuming output scaled between -1 and 1, scaled to [-128,128]

        # Convert LAB to uint8
        lab_color = lab_color.astype(np.float32)

        # Convert to RGB
        rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_Lab2RGB)
        rgb_color = np.clip(rgb_color, 0, 255).astype(np.uint8)

        # Show colorized image
        st.image(rgb_color, caption="Colorized Image", use_column_width=True)

        # Download button
        from PIL import Image
        colorized_pil = Image.fromarray(rgb_color)
        buf = np.array(colorized_pil)
        st.download_button(
            label="Download Colorized Image",
            data=colorized_pil.tobytes(),
            file_name="colorized_result.png",
            mime="image/png"
        )
