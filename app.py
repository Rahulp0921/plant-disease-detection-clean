import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    return load_model("plant_model.h5")

model = load_my_model()

input_shape = model.input_shape
img_size = input_shape[1]

# =========================
# LEAF DETECTION
# =========================
def is_leaf_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / mask.size

    return green_ratio > 0.10


# =========================
# GRAD-CAM HEATMAP
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed


# =========================
# LOAD CLASSES
# =========================
with open("classes.json", "r") as f:
    raw_class_names = json.load(f)

class_names = [name.replace("___", " ").replace("_", " ") for name in raw_class_names]

# =========================
# REMEDIES
# =========================
remedies = {
    "Pepper bell Bacterial spot": "Use disease-free seeds. Avoid overhead watering.",
    "Pepper bell healthy": "Plant is healthy. Maintain care.",

    "Potato Early blight": "Use fungicides and remove infected leaves.",
    "Potato Late blight": "Apply fungicides immediately.",
    "Potato healthy": "Healthy plant. Maintain care.",

    "Tomato Bacterial spot": "Use copper sprays and avoid overhead irrigation.",
    "Tomato Early blight": "Apply fungicide and remove infected leaves.",
    "Tomato Late blight": "Use resistant varieties and fungicides.",
    "Tomato Leaf Mold": "Improve airflow and reduce humidity.",
    "Tomato Septoria leaf spot": "Remove infected leaves and apply fungicide.",
    "Tomato Spider mites Two spotted spider mite": "Use neem oil or insecticidal soap.",
    "Tomato Target Spot": "Use fungicides and improve ventilation.",
    "Tomato Tomato YellowLeaf Curl Virus": "Control whiteflies.",
    "Tomato Tomato mosaic virus": "Remove infected plants.",
    "Tomato healthy": "Plant is healthy. Maintain care."
}

# =========================
# UI
# =========================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload or capture a plant leaf image")

option = st.radio("📥 Select Input Method", ["📁 Upload Image", "📸 Use Camera"])

img_file = None

if option == "📁 Upload Image":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
else:
    img_file = st.camera_input("Take Photo")

# =========================
# PROCESS IMAGE
# =========================
if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="📷 Input Image", use_column_width=True)

    try:
        # Leaf detection
        if not is_leaf_image(img):
            st.error("❌ Not a plant leaf. Please upload a leaf image.")
            st.stop()

        original_img = img.copy()

        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        label = class_names[class_id]

        st.markdown("## 🌿 Prediction Result")
        st.success(f"🌱 {label}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")

        # Top 3
        st.write("### 🔍 Top Predictions")
        top3 = np.argsort(pred[0])[-3:][::-1]
        for i in top3:
            st.write(f"{class_names[i]}: {pred[0][i]*100:.2f}%")

        # Heatmap
        st.write("### 🔥 Model Focus (Heatmap)")
        last_conv_layer = [layer.name for layer in model.layers if "conv" in layer.name][-1]

        heatmap = make_gradcam_heatmap(img, model, last_conv_layer)
        heatmap_img = overlay_heatmap(original_img, heatmap)

        st.image(heatmap_img, caption="Model Attention Area", use_column_width=True)

        # Remedy
        st.write("### 🌿 Remedy")
        if label in remedies:
            st.success(remedies[label])
        else:
            st.info("Consult expert.")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.text(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Final Year Project | AI Plant Disease Detection 🌱")