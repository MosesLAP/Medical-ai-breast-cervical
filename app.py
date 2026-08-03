import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import base64
import os

st.set_page_config(page_title="Breast Cancer AI Detection System", page_icon="🎗️", layout="wide")

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
.stApp {
    background: #0f1420;
    color: #e8e8ec;
}
section[data-testid="stSidebar"] {
    background: #151b2b;
    border-right: 1px solid #2a3348;
}
.main-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0px;
}
.subtitle {
    color: #8b93a8;
    font-size: 0.9rem;
    margin-bottom: 20px;
}
.stat-card {
    background: linear-gradient(145deg, #1a2236, #161d2e);
    border: 1px solid #2a3348;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 10px;
}
.stat-label { color: #8b93a8; font-size: 0.82rem; }
.stat-value { font-size: 1.7rem; font-weight: 800; color: #ffffff; }
.stat-sub { font-size: 0.78rem; color: #4ade80; }
.result-card-malignant {
    background: linear-gradient(145deg, #2a1420, #1f1018);
    border-left: 4px solid #f43f5e;
    border-radius: 12px;
    padding: 20px;
}
.result-card-benign {
    background: linear-gradient(145deg, #12261c, #0e1c16);
    border-left: 4px solid #4ade80;
    border-radius: 12px;
    padding: 20px;
}
.result-label-malignant { color: #f43f5e; font-size: 1.4rem; font-weight: 800; }
.result-label-benign { color: #4ade80; font-size: 1.4rem; font-weight: 800; }
.confidence-text { color: #ffffff; font-size: 1.1rem; font-weight: 700; }
.panel {
    background: #151b2b;
    border: 1px solid #2a3348;
    border-radius: 14px;
    padding: 20px;
}
.disclaimer-box {
    background: #2a2412;
    border: 1px solid #4d4520;
    border-radius: 10px;
    padding: 12px 16px;
    color: #e0c068;
    font-size: 0.82rem;
    margin-top: 14px;
}
.session-item {
    background: #1a2236;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 3px solid #565f78;
}
.hero-panel {
    background: linear-gradient(180deg, #1a2236, #0f1420);
    border: 1px solid #2a3348;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
    text-align: center;
}
.hero-quote {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.3;
    margin-bottom: 12px;
}
.hero-quote-highlight { color: #f43f5e; }
.hero-subtext {
    color: #8b93a8;
    font-size: 0.9rem;
    margin-bottom: 18px;
}
.hero-img {
    width: 100%;
    border-radius: 12px;
    margin-top: 10px;
}
.hero-icon-fallback {
    font-size: 4rem;
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPERS
# ============================================
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================
# MODELS
# ============================================
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

@st.cache_resource
def load_ultrasound_model():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ])
    base_model = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3), include_top=False, weights=None
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

histo_model = load_histo_model()
us_model = load_ultrasound_model()
ULTRASOUND_THRESHOLD = 0.35

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown('<div class="main-title">🎗️ Breast Cancer<br>AI Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered histopathology &amp; ultrasound analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔬 Histopathology Detection", "📡 Ultrasound Detection", "📈 Model Performance", "ℹ️ About & Disclaimer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div class="stat-card">
    <span style="color:#4ade80;">● Online</span><br>
    <span class="stat-label">Models: 2 deployed</span><br>
    <span class="stat-label">Built by Solomon Moses</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# DASHBOARD PAGE
# ============================================
if page == "📊 Dashboard":
    main_col, hero_col = st.columns([3, 1])

    with main_col:
        st.markdown('<div class="main-title">Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Overview of deployed models and real training results</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="stat-card"><div class="stat-label">Histopathology Accuracy</div><div class="stat-value">80.8%</div><div class="stat-sub">ROC-AUC 0.885</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stat-card"><div class="stat-label">Ultrasound Accuracy</div><div class="stat-value">77.6%</div><div class="stat-sub">ROC-AUC 0.827</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="stat-card"><div class="stat-label">Histopathology Images</div><div class="stat-value">16,000</div><div class="stat-sub">Real patient patches</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="stat-card"><div class="stat-label">Ultrasound Images</div><div class="stat-value">2,522</div><div class="stat-sub">BUSI + BUS-BRA datasets</div></div>', unsafe_allow_html=True)

        st.markdown("###")
        st.markdown('<div class="panel"><b>This Session\'s Predictions</b>', unsafe_allow_html=True)
        if len(st.session_state.history) == 0:
            st.markdown('<p style="color:#8b93a8;">No predictions made yet this session. Try the Histopathology or Ultrasound Detection tabs.</p>', unsafe_allow_html=True)
        else:
            for item in reversed(st.session_state.history[-5:]):
                color = "#f43f5e" if item['result'] == 'MALIGNANT' else "#4ade80"
                st.markdown(f"""
                <div class="session-item">
                <b>{item['model']}</b> — <span style="color:{color};font-weight:700;">{item['result']}</span>
                ({item['confidence']:.1f}%) — {item['time']}
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with hero_col:
        img_b64 = get_image_base64("doctor.jpg")
        st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-quote">Early <span class="hero-quote-highlight">detection</span> saves lives</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="hero-subtext">Timely screening and accurate diagnosis can make all the difference.</div>',
            unsafe_allow_html=True
        )
        if img_b64:
            st.markdown(f'<img src="data:image/jpeg;base64,{img_b64}" class="hero-img">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hero-icon-fallback">🩺</div>', unsafe_allow_html=True)
            st.caption("Add a doctor.jpg file to your project folder to display a photo here")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# HISTOPATHOLOGY PAGE
# ============================================
elif page == "🔬 Histopathology Detection":
    st.markdown('<div class="main-title">Histopathology Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a breast histopathology image patch</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        histo_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="histo")
        if histo_file is not None:
            image = Image.open(histo_file).convert('L')
            st.image(image, caption="Uploaded Image", width=260)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if histo_file is not None:
            img_resized = image.resize((48, 48))
            img_array = np.array(img_resized) / 255.0
            img_array = img_array.reshape(1, 48, 48, 1)

            with st.spinner("Analyzing tissue pattern..."):
                pred = histo_model.predict(img_array)[0][0]

            is_malignant = pred > 0.5
            confidence = pred * 100 if is_malignant else (1 - pred) * 100
            label = "MALIGNANT" if is_malignant else "BENIGN"

            st.session_state.history.append({
                'model': 'Histopathology', 'result': label,
                'confidence': confidence, 'time': datetime.now().strftime('%H:%M:%S')
            })

            css_class = "result-card-malignant" if is_malignant else "result-card-benign"
            label_class = "result-label-malignant" if is_malignant else "result-label-benign"
            st.markdown(f"""
            <div class="{css_class}">
            <div class="{label_class}">{label}</div>
            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("###### Probability Distribution")
            st.progress(float(pred))
            st.caption(f"Benign: {(1-pred)*100:.1f}%  |  Malignant: {pred*100:.1f}%")
            st.markdown('<div class="disclaimer-box">⚕️ Research prototype for educational purposes — not a diagnostic tool.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel" style="color:#8b93a8;">Upload an image to see prediction results here.</div>', unsafe_allow_html=True)

# ============================================
# ULTRASOUND PAGE
# ============================================
elif page == "📡 Ultrasound Detection":
    st.markdown('<div class="main-title">Ultrasound Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a breast ultrasound image</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        us_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="ultrasound")
        if us_file is not None:
            image = Image.open(us_file).convert('L')
            st.image(image, caption="Uploaded Image", width=260)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if us_file is not None:
            img_resized = image.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_rgb = np.repeat(img_array[:, :, np.newaxis], 3, axis=-1).reshape(1, 128, 128, 3)

            with st.spinner("Analyzing ultrasound image..."):
                pred = us_model.predict(img_rgb)[0][0]

            is_malignant = pred > ULTRASOUND_THRESHOLD
            confidence = pred * 100 if is_malignant else (1 - pred) * 100
            label = "MALIGNANT" if is_malignant else "BENIGN"

            st.session_state.history.append({
                'model': 'Ultrasound', 'result': label,
                'confidence': confidence, 'time': datetime.now().strftime('%H:%M:%S')
            })

            css_class = "result-card-malignant" if is_malignant else "result-card-benign"
            label_class = "result-label-malignant" if is_malignant else "result-label-benign"
            st.markdown(f"""
            <div class="{css_class}">
            <div class="{label_class}">{label}</div>
            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("###### Probability Distribution")
            st.progress(float(pred))
            st.caption(f"Benign: {(1-pred)*100:.1f}%  |  Malignant: {pred*100:.1f}%")
            st.markdown(
                '<div class="disclaimer-box">⚕️ Research prototype — not a diagnostic tool. '
                'Findings from ultrasound are often correlated with additional imaging as part of standard diagnostic workflow.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="panel" style="color:#8b93a8;">Upload an image to see prediction results here.</div>', unsafe_allow_html=True)

# ============================================
# MODEL PERFORMANCE PAGE
# ============================================
elif page == "📈 Model Performance":
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real evaluation metrics from held-out test sets</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="panel">
        <b>Histopathology CNN</b><br><br>
        Accuracy: <b>80.8%</b><br>
        ROC-AUC: <b>0.885</b><br>
        Sensitivity: <b>85.0%</b><br>
        Specificity: <b>76.6%</b><br>
        Training images: <b>16,000</b>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="panel">
        <b>Ultrasound CNN (Transfer Learning)</b><br><br>
        Accuracy: <b>77.6%</b><br>
        ROC-AUC: <b>0.827</b><br>
        Sensitivity: <b>72.6%</b><br>
        Specificity: <b>80.1%</b><br>
        Training images: <b>2,522</b>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# ABOUT PAGE
# ============================================
elif page == "ℹ️ About & Disclaimer":
    st.markdown('<div class="main-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
    This is a research prototype demonstrating deep learning approaches to breast cancer image
    classification across two modalities: histopathology and ultrasound. Both models were trained
    on real, publicly available medical imaging datasets.
    <br><br>
    <b>Important:</b> This tool is for educational and research demonstration purposes only.
    It is not a certified medical device and must not be used for actual clinical diagnosis or
    treatment decisions. Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#4a5268; font-size:0.8rem; margin-top:40px; padding-top:15px; border-top:1px solid #2a3348;">
Built by Solomon Moses — Federal University of Health Sciences, Azare, Nigeria
</div>
""", unsafe_allow_html=True)