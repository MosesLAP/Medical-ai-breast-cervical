import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="Breast Cancer Detection Prototype", page_icon="🎗️", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #fdf6f9 0%, #f3e9f0 40%, #eef4f7 100%); background-attachment: fixed; }
.main-title { font-size: 2.4rem; font-weight: 800; color: #7a2e5c; text-align: center; margin-bottom: 0px; }
.subtitle { text-align: center; color: #5c5c6e; font-size: 1.05rem; margin-top: 4px; margin-bottom: 25px; }
.card { background: rgba(255, 255, 255, 0.75); border-radius: 16px; padding: 25px 30px; box-shadow: 0 4px 20px rgba(122, 46, 92, 0.12); margin-bottom: 20px; }
.result-malignant { background: linear-gradient(90deg, #ffe1e1, #ffcccc); border-left: 6px solid #d63447; border-radius: 10px; padding: 18px 20px; font-size: 1.1rem; font-weight: 600; color: #7a1020; }
.result-benign { background: linear-gradient(90deg, #e1ffe6, #ccf5d6); border-left: 6px solid #2e9e5b; border-radius: 10px; padding: 18px 20px; font-size: 1.1rem; font-weight: 600; color: #1c5c36; }
.footer { text-align: center; color: #8a8a9a; font-size: 0.85rem; margin-top: 40px; padding-top: 15px; border-top: 1px solid #e0d5dc; }
.disclaimer { text-align: center; color: #a37a4d; font-size: 0.8rem; background: #fff8ec; border-radius: 8px; padding: 8px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎗️ Breast Cancer Histopathology Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-assisted analysis of breast tissue histopathology images</div>', unsafe_allow_html=True)

# Rebuild the EXACT same architecture used in training
@st.cache_resource
def load_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('cancer_cnn_final.weights.h5')
    return model

model = load_model()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader("Choose a histopathology image patch (PNG or JPG)", type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.image(image, caption="Uploaded Image", width=220)

    img_resized = image.resize((48, 48))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    with st.spinner("Analyzing tissue pattern..."):
        prediction = model.predict(img_array)[0][0]

    with col2:
        st.markdown("### Result")
        if prediction > 0.5:
            confidence = prediction * 100
            st.markdown(f'<div class="result-malignant">⚠️ Prediction: MALIGNANT<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
        else:
            confidence = (1 - prediction) * 100
            st.markdown(f'<div class="result-benign">✅ Prediction: BENIGN<br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)

        st.progress(float(prediction if prediction > 0.5 else 1 - prediction))

    st.markdown(
        '<div class="disclaimer">⚕️ This is a research prototype for educational purposes only. '
        'Not intended for clinical diagnosis or medical decision-making.</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="footer">Built by <b>Solomon Moses</b><br>Breast Cancer Detection Prototype — '
    'CNN trained on histopathology images</div>',
    unsafe_allow_html=True
)