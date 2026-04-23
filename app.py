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

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload or capture a leaf image to detect disease")

# ================================
# IMAGE INPUT (UPLOAD + CAMERA)
# ================================

img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
camera_img = st.camera_input("Or take a photo")

# Use camera if available
if camera_img is not None:
    img_file = camera_img

# ================================
# PROCESS IMAGE
# ================================

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        # Preprocessing (FIXED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')

        # Prediction
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        # ================================
        # OUTPUT
        # ================================

        st.success(f"🌱 Prediction: {class_names[class_id]}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # Top 3 predictions (PRO FEATURE)
        st.write("### 🔍 Top Predictions")
        top3 = np.argsort(pred[0])[-3:][::-1]

        for i in top3:
            st.write(f"{class_names[i]}: {pred[0][i]*100:.2f}%")

        # Low confidence warning
        if confidence < 0.7:
            st.warning("⚠️ Low confidence. Try clearer image or better lighting.")

    except Exception as e:
        st.error("Error during prediction. Please try another image.")
        st.text(str(e))