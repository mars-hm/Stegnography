import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "D:\M.Sc CS\Deep Learning\Stegnography\Deep_steganography_model.h5"
model = load_model(MODEL_PATH)

# Define image size (64x64 assumed)
IMG_SHAPE = (64, 64)

# Function to preprocess image
def preprocess_image(image: Image.Image):
    """
    Preprocess an image for model input:
    - Resize to required dimensions
    - Normalize pixel values to [0, 1]
    - Add batch dimension
    """
    image = image.resize(IMG_SHAPE)
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit App
st.title("Deep Learning Steganography Model")
st.write(
    "This application demonstrates a deep learning model for steganography. "
    "Upload a **cover image** and a **secret image** for encoding or decoding."
)

# Upload the cover image
cover_file = st.file_uploader("Upload the Cover Image", type=["jpg", "png", "jpeg"])

# Upload the secret image
secret_file = st.file_uploader("Upload the Secret Image", type=["jpg", "png", "jpeg"])

if cover_file and secret_file:
    # Display the cover image
    cover_image = Image.open(cover_file).convert("RGB")
    st.image(cover_image, caption="Cover Image", use_column_width=True)

    # Display the secret image
    secret_image = Image.open(secret_file).convert("RGB")
    st.image(secret_image, caption="Secret Image", use_column_width=True)

    # Preprocess both images
    cover_preprocessed = preprocess_image(cover_image)
    secret_preprocessed = preprocess_image(secret_image)

    # Predict using the model
    st.write("Processing the images...")
    try:
        prediction = model.predict([cover_preprocessed, secret_preprocessed])
        st.write("Prediction result:", prediction)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

elif cover_file:
    st.warning("Please upload both a cover image and a secret image.")
else:
    st.info("Please upload a cover image to begin.")

# Sidebar Information
st.sidebar.title("About")
st.sidebar.info(
    """
    This application is a demonstration of a deep learning-based steganography model. 
    The project was developed by:

    - **Soumya Pal**: Roll No: 231043, Email: soumya.cs23@duk.ac.in
    - **Marcie M**: Roll No: 231030, Email: marcie.cs23@duk.ac.in
    """
)

