import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time

# Page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern medical theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1e40af 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.9), rgba(59, 130, 246, 0.8));
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        text-align: center;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1rem;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.2);
    }
    
    .tumor-card-glioma { border-left: 4px solid #ef4444; }
    .tumor-card-meningioma { border-left: 4px solid #f97316; }
    .tumor-card-pituitary { border-left: 4px solid #a855f7; }
    .tumor-card-notumor { border-left: 4px solid #22c55e; }
    
    .tumor-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .tumor-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .upload-zone {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1));
        border: 2px dashed rgba(96, 165, 250, 0.5);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-zone:hover {
        border-color: #60a5fa;
        box-shadow: 0 0 40px rgba(96, 165, 250, 0.3);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.5rem;
    }
    
    .prediction-main {
        font-size: 2rem;
        font-weight: 700;
        color: #22c55e;
        text-align: center;
    }
    
    .confidence-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 12px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }
    
    .bar-glioma { background: linear-gradient(90deg, #ef4444, #f87171); }
    .bar-meningioma { background: linear-gradient(90deg, #f97316, #fb923c); }
    .bar-pituitary { background: linear-gradient(90deg, #a855f7, #c084fc); }
    .bar-notumor { background: linear-gradient(90deg, #22c55e, #4ade80); }
    
    .about-section {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .bullet-point {
        color: #94a3b8;
        padding: 0.3rem 0;
        font-size: 0.95rem;
    }
    
    .disclaimer {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 2rem;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);
    }
    
    .stSpinner > div {
        border-color: #3b82f6 transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

IMG_SIZE = (150, 150)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS = {
    "glioma": "#ef4444",
    "meningioma": "#f97316", 
    "pituitary": "#a855f7",
    "notumor": "#22c55e"
}

TUMOR_INFO = {
    "glioma": {
        "icon": "🔴",
        "severity": "High",
        "description": "Arises from glial cells in the brain and spinal cord. Most common primary brain tumor.",
        "bullets": ["• Originates from glial cells", "• 30% of all brain tumors", "• Requires immediate attention", "• Various grades (I-IV)"]
    },
    "meningioma": {
        "icon": "🟠", 
        "severity": "Moderate",
        "description": "Develops from meninges, the protective membranes surrounding the brain and spinal cord.",
        "bullets": ["• Usually benign (90%)", "• Slow-growing tumor", "• Most common in adults", "• Often discovered incidentally"]
    },
    "pituitary": {
        "icon": "🟣",
        "severity": "Variable",
        "description": "Forms in the pituitary gland and can affect hormone production throughout the body.",
        "bullets": ["• Affects hormone levels", "• Usually non-cancerous", "• 10-15% of brain tumors", "• Treatable with medication"]
    },
    "notumor": {
        "icon": "🟢",
        "severity": "None",
        "description": "No tumor detected. The MRI scan shows normal brain tissue without abnormal growths.",
        "bullets": ["• Healthy brain tissue", "• No abnormalities detected", "• Regular checkups advised", "• Maintain healthy lifestyle"]
    }
}

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# Hero Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 Brain Tumor Classifier</div>
    <div class="subtitle">Advanced CNN-Powered Medical Image Analysis</div>
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-value">CNN</div>
            <div class="stat-label">Architecture</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">4</div>
            <div class="stat-label">Classes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">150×150</div>
            <div class="stat-label">Input Size</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">Real-time</div>
            <div class="stat-label">Prediction</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# About Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">👨‍⚕️ About This Project</div>
        <div class="tumor-desc">
            This deep learning application uses a Convolutional Neural Network (CNN) 
            trained on thousands of MRI scans to classify brain tumors into four categories.
            Designed for medical students, researchers, and healthcare professionals as an
            educational tool to understand AI-assisted diagnosis.
        </div>
        <div style="margin-top: 1rem;">
            <div class="bullet-point">🎓 Educational purpose only</div>
            <div class="bullet-point">🔬 Research-grade accuracy</div>
            <div class="bullet-point">⚡ Real-time classification</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">📊 Model Statistics</div>
        <div class="bullet-point">📈 <strong>Architecture:</strong> Convolutional Neural Network</div>
        <div class="bullet-point">🖼️ <strong>Input Size:</strong> 150 × 150 pixels (RGB)</div>
        <div class="bullet-point">🏷️ <strong>Classes:</strong> Glioma, Meningioma, Pituitary, No Tumor</div>
        <div class="bullet-point">📚 <strong>Dataset:</strong> Brain Tumor MRI Dataset</div>
        <div class="bullet-point">🎯 <strong>Output:</strong> Softmax probabilities</div>
        <div class="bullet-point">⚙️ <strong>Framework:</strong> TensorFlow / Keras</div>
    </div>
    """, unsafe_allow_html=True)

# Tumor Classes Grid
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title" style="text-align:center; font-size:1.8rem;">📋 Tumor Classification Categories</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="glass-card tumor-card-glioma">
        <div class="tumor-title">{TUMOR_INFO['glioma']['icon']} Glioma</div>
        <div style="color: #ef4444; font-size: 0.8rem; margin-bottom: 0.5rem;">Severity: {TUMOR_INFO['glioma']['severity']}</div>
        <div class="tumor-desc">{TUMOR_INFO['glioma']['description']}</div>
        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
            {'<br>'.join(TUMOR_INFO['glioma']['bullets'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card tumor-card-pituitary">
        <div class="tumor-title">{TUMOR_INFO['pituitary']['icon']} Pituitary</div>
        <div style="color: #a855f7; font-size: 0.8rem; margin-bottom: 0.5rem;">Severity: {TUMOR_INFO['pituitary']['severity']}</div>
        <div class="tumor-desc">{TUMOR_INFO['pituitary']['description']}</div>
        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
            {'<br>'.join(TUMOR_INFO['pituitary']['bullets'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-card tumor-card-meningioma">
        <div class="tumor-title">{TUMOR_INFO['meningioma']['icon']} Meningioma</div>
        <div style="color: #f97316; font-size: 0.8rem; margin-bottom: 0.5rem;">Severity: {TUMOR_INFO['meningioma']['severity']}</div>
        <div class="tumor-desc">{TUMOR_INFO['meningioma']['description']}</div>
        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
            {'<br>'.join(TUMOR_INFO['meningioma']['bullets'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card tumor-card-notumor">
        <div class="tumor-title">{TUMOR_INFO['notumor']['icon']} No Tumor</div>
        <div style="color: #22c55e; font-size: 0.8rem; margin-bottom: 0.5rem;">Severity: {TUMOR_INFO['notumor']['severity']}</div>
        <div class="tumor-desc">{TUMOR_INFO['notumor']['description']}</div>
        <div style="margin-top: 0.8rem; font-size: 0.85rem; color: #cbd5e1;">
            {'<br>'.join(TUMOR_INFO['notumor']['bullets'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Upload Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title" style="text-align:center; font-size:1.8rem;">🔬 Analyze MRI Scan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-zone">
    <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
    <div style="color: #94a3b8; font-size: 1.1rem;">Drag and drop your MRI scan or click to browse</div>
    <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">Supports PNG, JPG, JPEG</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div class="section-title" style="justify-content: center;">🖼️ Uploaded MRI Scan</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; height: 100%;">
            <div class="section-title" style="justify-content: center;">🎯 Analysis Results</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔮 Analyze MRI Scan", type="primary"):
            with st.spinner("🧠 AI is analyzing the scan..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                class_name = CLASS_NAMES[predicted_class]
                
                progress_bar.empty()
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="prediction-main">
                        {TUMOR_INFO[class_name]['icon']} {class_name.upper()}
                    </div>
                    <div style="text-align: center; color: #94a3b8; margin-top: 0.5rem;">
                        Confidence: <strong style="color: #22c55e;">{confidence:.1%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div style="color: #94a3b8; font-weight: 600; margin-bottom: 1rem;">📊 All Class Probabilities</div>', unsafe_allow_html=True)
                
                for i, (cls, score) in enumerate(zip(CLASS_NAMES, prediction[0])):
                    bar_class = f"bar-{cls}"
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; color: #e2e8f0; margin-bottom: 0.3rem;">
                            <span>{TUMOR_INFO[cls]['icon']} {cls.capitalize()}</span>
                            <span style="color: {CLASS_COLORS[cls]};">{score:.1%}</span>
                        </div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar {bar_class}" style="width: {score*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is designed for educational and research purposes only. 
    It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified 
    healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 3rem; padding: 2rem;">
    <div>Built by Nyla</div>
    <div style="font-size: 0.8rem; margin-top: 0.5rem;">© 2024 Brain Tumor Classifier | For Educational Use Only</div>
</div>
""", unsafe_allow_html=True)

