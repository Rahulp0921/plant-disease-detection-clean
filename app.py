import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2e7d32;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model("plant_model.h5")

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Potato___Early_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

# Remedies
remedies = {
    'Tomato___Early_blight': "Remove infected leaves and apply fungicide.",
    'Tomato___Late_blight': "Use copper-based fungicide and improve drainage.",
    'Tomato___healthy': "Plant is healthy. Maintain proper care.",
    'Potato___Early_blight': "Use resistant seeds and apply fungicide.",
    'Potato___healthy': "Healthy crop. Maintain irrigation.",
    'Pepper__bell___Bacterial_spot': "Use disease-free seeds and apply copper spray."
}

# Title
st.markdown('<div class="title">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)

st.write("Upload a plant leaf image and detect diseases instantly.")

# Upload
file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Preprocess
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, 128, 128, 3)

    # Predict
    pred = model.predict(img_resized)
    index = np.argmax(pred)

    disease = class_names[index]
    confidence = float(np.max(pred)) * 100

    display_name = disease.replace("___", " - ").replace("_", " ")

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🌱 Prediction")
        st.write(f"**{display_name}**")

        st.subheader("📊 Confidence")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}%")

        st.subheader("🌿 Remedy")
        st.success(remedies.get(disease, "No remedy available."))

        st.markdown('</div>', unsafe_allow_html=True)

# Optional graph
try:
    st.markdown("### 📈 Model Performance")
    st.image("accuracy_graph.png")
except:
    pass