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

# =========================
# LOAD CLASSES
# =========================
with open("classes.json", "r") as f:
    raw_class_names = json.load(f)

# Clean class names
class_names = [name.replace("___", " ").replace("_", " ") for name in raw_class_names]

# =========================
# REMEDIES (ALL CLASSES)
# =========================
remedies = {
    "Pepper bell Bacterial spot": "Use disease-free seeds. Avoid overhead watering. Apply copper-based bactericides.",
    "Pepper bell healthy": "Plant is healthy. Maintain proper watering and sunlight.",

    "Potato Early blight": "Remove infected leaves. Use fungicides like chlorothalonil.",
    "Potato Late blight": "Apply fungicides immediately. Destroy infected plants to prevent spread.",
    "Potato healthy": "Plant is healthy. Maintain proper care and soil health.",

    "Tomato Bacterial spot": "Use certified seeds. Avoid overhead irrigation. Apply copper sprays.",
    "Tomato Early blight": "Use crop rotation. Apply fungicide. Remove infected leaves.",
    "Tomato Late blight": "Use resistant varieties. Apply fungicides. Avoid wet conditions.",
    "Tomato Leaf Mold": "Improve air circulation. Reduce humidity. Avoid overcrowding.",
    "Tomato Septoria leaf spot": "Remove infected leaves. Use fungicides. Avoid wet leaves.",
    "Tomato Spider mites Two spotted spider mite": "Use insecticidal soap or neem oil. Maintain humidity.",
    "Tomato Target Spot": "Remove infected leaves. Use fungicides. Improve ventilation.",
    "Tomato Tomato YellowLeaf Curl Virus": "Control whiteflies. Remove infected plants immediately.",
    "Tomato Tomato mosaic virus": "Remove infected plants. Disinfect tools. Avoid handling plants when wet.",
    "Tomato healthy": "Plant is healthy. Maintain proper watering, sunlight, and nutrients."
}

# =========================
# UI
# =========================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload or capture a plant leaf image to detect diseases using AI.")

# Input
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
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)

        label = class_names[class_id]

        # =========================
        # OUTPUT
        # =========================
        st.markdown("## 🌿 Prediction Result")

        st.success(f"🌱 Prediction: {label}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # Top 3 predictions
        st.write("### 🔍 Top Predictions")
        top3 = np.argsort(pred[0])[-3:][::-1]

        for i in top3:
            st.write(f"{class_names[i]}: {pred[0][i]*100:.2f}%")

        # Confidence warning
        if confidence < 0.85:
            st.warning("⚠️ Low confidence. Try a clearer image or better lighting.")

        # =========================
        # REMEDY SECTION
        # =========================
        st.write("### 🌿 Remedy")

        if label in remedies:
            st.success(remedies[label])
        else:
            st.info("General advice: Remove affected leaves and consult an agricultural expert.")

        # Accuracy note
        st.caption("Model accuracy: ~92% on validation dataset")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.text(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Developed as Final Year Project | AI-Based Plant Disease Detection 🌱")