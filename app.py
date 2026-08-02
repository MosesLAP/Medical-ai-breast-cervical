import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="Breast Cancer Detection Suite", page_icon="🎗️", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #fdf6f9 0%, #f3e9f0 40%, #eef4f7 100%); background-attachment: fixed; }
.main-title { font-size: 2.2rem; font-weight: 800; color: #7a2e5c; text-align: center; margin-bottom: 0px; }
.subtitle { text-align: center; color: #5c5c6e; font-size: 1rem; margin-top: 4px; margin-bottom: 20px; }
.result-malignant { background: linear-gradient(90deg, #ffe1e1, #ffcccc); border-left: 6px solid #d63447; border-radius: 10px; padding: 18px 20px; font-size: 1.1rem; font-weight: 600; color: #7a1020; }
.result-benign { background: linear-gradient(90deg, #e1ffe6, #ccf5d6); border-left: 6px solid #2e9e5b; border-radius: 10px; padding: 18px 20px; font-size: 1.1rem; font-weight: 600; color: #1c5c36; }
.disclaimer { text-align: center; color: #a37a4d; font-size: 0.8rem; background: #fff8ec; border-radius: 8px; padding: 8px; margin-top: 15px; }
.footer { text-align: center; color: #8a8a9a; font-size: 0.85rem; margin-top: 40px; padding-top: 15px; border-top: 1px solid #e0d5dc; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎗️ Breast Cancer Detection Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-modality AI-assisted breast cancer screening support</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔬 Histopathology", "📡 Ultrasound"])

# ============================================
# TAB 1: HISTOPATHOLOGY
# ============================================
with tab1:
    @st.cache_resource
    def load_histo_model():
        m = keras.Sequential([
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
        m.load_weights('cancer_cnn_final.weights.h5')
        return m

    histo_model = load_histo_model()

    st.markdown("### Upload a Histopathology Image")
    histo_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="histo")

    if histo_file is not None:
        image = Image.open(histo_file).convert('L')
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(image, caption="Uploaded Image", width=200)

        img_resized = image.resize((48, 48))
        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, 48, 48, 1)

        with st.spinner("Analyzing..."):
            pred = histo_model.predict(img_array)[0][0]

        with col2:
            st.markdown("### Result")
            if pred > 0.5:
                st.markdown(f'<div class="result-malignant">⚠️ MALIGNANT<br>Confidence: {pred*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-benign">✅ BENIGN<br>Confidence: {(1-pred)*100:.1f}%</div>', unsafe_allow_html=True)
            st.progress(float(pred if pred > 0.5 else 1-pred))

        st.markdown('<div class="disclaimer">⚕️ Research prototype — not a diagnostic tool.</div>', unsafe_allow_html=True)

# ============================================
# TAB 2: ULTRASOUND (rebuilt architecture + weights-only loading)
# ============================================
with tab2:
    ULTRASOUND_THRESHOLD = 0.35

    @st.cache_resource
    def load_ultrasound_model():
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
        ])

        base_model = keras.applications.MobileNetV2(
            input_shape=(128, 128, 3),
            include_top=False,
            weights=None  # we'll load our own trained weights, not ImageNet defaults
        )

        inputs = keras.Input(shape=(128, 128, 3))
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        m = keras.Model(inputs, outputs)
        m.load_weights('ultrasound_v2.weights.h5')
        return m

    us_model = load_ultrasound_model()

    st.markdown("### Upload a Breast Ultrasound Image")
    us_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="ultrasound")

    if us_file is not None:
        image = Image.open(us_file).convert('L')
        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.image(image, caption="Uploaded Image", width=200)

        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_rgb = np.repeat(img_array[:, :, np.newaxis], 3, axis=-1)
        img_rgb = img_rgb.reshape(1, 128, 128, 3)

        with st.spinner("Analyzing..."):
            pred = us_model.predict(img_rgb)[0][0]

        with col2:
            st.markdown("### Result")
            if pred > ULTRASOUND_THRESHOLD:
                st.markdown(f'<div class="result-malignant">⚠️ MALIGNANT<br>Confidence: {pred*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-benign">✅ BENIGN<br>Confidence: {(1-pred)*100:.1f}%</div>', unsafe_allow_html=True)
            st.progress(float(pred if pred > ULTRASOUND_THRESHOLD else 1-pred))

        st.markdown(
            '<div class="disclaimer">⚕️ Research prototype — not a diagnostic tool. '
            'Findings from ultrasound are often correlated with additional imaging (e.g., mammography or '
            'biopsy) as part of standard diagnostic workflow.</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="footer">Built by <b>Solomon Moses</b><br>'
    'Multi-Modality Breast Cancer Detection Suite — Histopathology &amp; Ultrasound CNNs</div>',
    unsafe_allow_html=True
)