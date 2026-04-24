import streamlit as st
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model

# Load model
@st.cache_resource
def load_my_model():
    return load_model("plant_model.h5")

model = load_my_model()

# 🔥 Get correct input size from model
input_shape = model.input_shape
img_size = input_shape[1]

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

st.title("🌿 Plant Disease Detection")

img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_img = st.camera_input("Take Photo")

if camera_img is not None:
    img_file = camera_img

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Input Image")

    try:
        # 🔥 Use model's expected size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype("float32")

        # Debug
        st.write("Model expects:", input_shape)
        st.write("Image shape:", img.shape)

        # Predict
        pred = model.predict(img, verbose=0)

        class_id = np.argmax(pred)
        confidence = np.max(pred)

        st.success(f"Prediction: {class_names[class_id]}")
        st.write(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error("Prediction failed")
        st.text(str(e))