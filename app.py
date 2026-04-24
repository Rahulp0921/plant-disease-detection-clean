import streamlit as st
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    return load_model("plant_model.h5")

model = load_my_model()

# Get model input size
input_shape = model.input_shape
img_size = input_shape[1]

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# =========================
# UI
# =========================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload or capture a plant leaf image for disease detection")

# Image input
img_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])
camera_img = st.camera_input("📸 Take Photo")

if camera_img is not None:
    img_file = camera_img

# =========================
# PROCESS IMAGE
# =========================
if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="📷 Input Image", use_column_width=True)

    try:
        # Preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        # =========================
        # OUTPUT
        # =========================
        st.markdown("## 🌿 Prediction Result")

        st.success(f"🌱 Prediction: {class_names[class_id]}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # Top 3 predictions
        st.write("### 🔍 Top Predictions")
        top3 = np.argsort(pred[0])[-3:][::-1]

        for i in top3:
            st.write(f"{class_names[i]}: {pred[0][i]*100:.2f}%")

        # Confidence warning
        if confidence < 0.85:
            st.warning("⚠️ Low confidence. Try a clearer image or better lighting.")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.text(str(e))