import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time

# Page config
st.set_page_config(
    page_title="Neural MRI Analysis",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Sophisticated Teal/Slate Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background: linear-gradient(160deg, #0a0f0f 0%, #0d1b1b 40%, #112020 100%);
    }
    
    /* Navigation Bar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(13, 27, 27, 0.8);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(94, 234, 212, 0.1);
        margin: -1rem -1rem 2rem -1rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nav-logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #14b8a6, #5eead4);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        color: #0a0f0f;
        font-weight: 700;
    }
    
    .nav-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f0fdfa;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-link {
        color: #5eead4;
        font-size: 0.9rem;
        font-weight: 500;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .nav-link:hover {
        background: rgba(94, 234, 212, 0.1);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        margin-bottom: 3rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(94, 234, 212, 0.1);
        border: 1px solid rgba(94, 234, 212, 0.3);
        color: #5eead4;
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        color: #f0fdfa;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .hero-title span {
        background: linear-gradient(135deg, #14b8a6, #5eead4, #99f6e4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: #6b7280;
        font-size: 1.25rem;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto 2rem;
        line-height: 1.6;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 4rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(94, 234, 212, 0.1);
    }
    
    .hero-stat {
        text-align: center;
    }
    
    .hero-stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #5eead4;
        line-height: 1;
    }
    
    .hero-stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .section-tag {
        color: #14b8a6;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.25rem;
        font-weight: 600;
        color: #f0fdfa;
        margin-bottom: 0.5rem;
    }
    
    .section-desc {
        color: #6b7280;
        font-size: 1rem;
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(20, 30, 30, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(94, 234, 212, 0.08);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1.5rem;
    }
    
    .glass-card:hover {
        transform: translateY(-6px);
        border-color: rgba(94, 234, 212, 0.2);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4), 0 0 60px rgba(94, 234, 212, 0.05);
    }
    
    /* Tumor Cards */
    .tumor-card {
        position: relative;
        overflow: hidden;
    }
    
    .tumor-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        border-radius: 3px 0 0 3px;
    }
    
    .tumor-card-glioma::before { background: linear-gradient(180deg, #f43f5e, #e11d48); }
    .tumor-card-meningioma::before { background: linear-gradient(180deg, #f59e0b, #d97706); }
    .tumor-card-pituitary::before { background: linear-gradient(180deg, #a855f7, #7c3aed); }
    .tumor-card-notumor::before { background: linear-gradient(180deg, #10b981, #059669); }
    
    .tumor-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .tumor-icon-glioma { background: rgba(244, 63, 94, 0.15); color: #f43f5e; }
    .tumor-icon-meningioma { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .tumor-icon-pituitary { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
    .tumor-icon-notumor { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    
    .tumor-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #f0fdfa;
        margin-bottom: 0.25rem;
    }
    
    .tumor-severity {
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1rem;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        display: inline-block;
    }
    
    .severity-high { background: rgba(244, 63, 94, 0.15); color: #f43f5e; }
    .severity-moderate { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .severity-variable { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
    .severity-none { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    
    .tumor-desc {
        color: #9ca3af;
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    
    .tumor-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tumor-list li {
        color: #6b7280;
        font-size: 0.85rem;
        padding: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .tumor-list li::before {
        content: '→';
        color: #5eead4;
        font-weight: 600;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    
    .feature-card {
        background: rgba(20, 30, 30, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(94, 234, 212, 0.05);
    }
    
    .feature-icon {
        width: 40px;
        height: 40px;
        background: rgba(94, 234, 212, 0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #5eead4;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: #f0fdfa;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #6b7280;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    /* Upload Section */
    .upload-container {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.05), rgba(94, 234, 212, 0.02));
        border: 2px dashed rgba(94, 234, 212, 0.2);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 2rem 0;
    }
    
    .upload-container:hover {
        border-color: rgba(94, 234, 212, 0.4);
        box-shadow: 0 0 60px rgba(94, 234, 212, 0.1);
    }
    
    .upload-icon {
        width: 80px;
        height: 80px;
        background: rgba(94, 234, 212, 0.1);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        font-size: 2rem;
        color: #5eead4;
    }
    
    .upload-title {
        color: #f0fdfa;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .upload-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    .upload-formats {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }
    
    .format-badge {
        background: rgba(94, 234, 212, 0.1);
        color: #5eead4;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Results Section */
    .result-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
    }
    
    .result-header {
        text-align: center;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(94, 234, 212, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .result-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .result-confidence {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .result-confidence span {
        color: #5eead4;
        font-weight: 600;
    }
    
    /* Probability Bars */
    .prob-item {
        margin-bottom: 1.25rem;
    }
    
    .prob-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .prob-name {
        color: #e5e7eb;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .prob-value {
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    
    .prob-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .prob-bar-glioma { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .prob-bar-meningioma { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .prob-bar-pituitary { background: linear-gradient(90deg, #a855f7, #c084fc); }
    .prob-bar-notumor { background: linear-gradient(90deg, #10b981, #34d399); }
    
    /* Disclaimer */
    .disclaimer {
        background: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-top: 3rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .disclaimer-icon {
        width: 24px;
        height: 24px;
        background: rgba(245, 158, 11, 0.2);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f59e0b;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
        margin-top: 2px;
    }
    
    .disclaimer-text {
        color: #9ca3af;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .disclaimer-text strong {
        color: #f59e0b;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(94, 234, 212, 0.08);
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 1.5rem;
    }
    
    .footer-link {
        color: #6b7280;
        font-size: 0.9rem;
        transition: color 0.3s ease;
    }
    
    .footer-link:hover {
        color: #5eead4;
    }
    
    .footer-copyright {
        color: #4b5563;
        font-size: 0.85rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(94, 234, 212, 0.2), transparent);
        margin: 4rem 0;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #14b8a6, #0d9488);
        color: #0a0f0f;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(20, 184, 166, 0.3);
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
    "glioma": "#f43f5e",
    "meningioma": "#f59e0b",
    "pituitary": "#a855f7",
    "notumor": "#10b981"
}

TUMOR_INFO = {
    "glioma": {
        "icon": "G",
        "severity": "High Risk",
        "severity_class": "severity-high",
        "description": "Originates from glial cells in the brain and spinal cord. Represents approximately 30% of all brain tumors and 80% of malignant brain tumors.",
        "points": ["Originates from glial cells", "Most common primary brain tumor", "Requires immediate clinical attention", "Classified in grades I through IV"]
    },
    "meningioma": {
        "icon": "M",
        "severity": "Moderate",
        "severity_class": "severity-moderate",
        "description": "Develops from the meninges, the protective membranes surrounding the brain and spinal cord. Usually grows slowly.",
        "points": ["Typically benign in 90% of cases", "Slow-growing neoplasm", "Most common in adults 40-70", "Often discovered incidentally"]
    },
    "pituitary": {
        "icon": "P",
        "severity": "Variable",
        "severity_class": "severity-variable",
        "description": "Forms in the pituitary gland and can significantly affect hormone production throughout the body.",
        "points": ["Affects hormonal balance", "Usually non-cancerous growth", "Accounts for 10-15% of brain tumors", "Often treatable with medication"]
    },
    "notumor": {
        "icon": "N",
        "severity": "Normal",
        "severity_class": "severity-none",
        "description": "No tumor detected in the scan. The MRI shows normal brain tissue without any abnormal growths or masses.",
        "points": ["Healthy brain tissue observed", "No abnormalities detected", "Continue regular checkups", "Maintain healthy lifestyle"]
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

# Navigation
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-icon">◉</div>
        <div class="nav-title">NeuroScan AI</div>
    </div>
    <div class="nav-links">
        <span class="nav-link">Overview</span>
        <span class="nav-link">Analysis</span>
        <span class="nav-link">Research</span>
        <span class="nav-link">Documentation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">◈ Advanced Neural Network Classification</div>
    <h1 class="hero-title">Brain Tumor <span>Classification</span></h1>
    <p class="hero-subtitle">
        Leveraging deep convolutional neural networks for accurate MRI-based 
        brain tumor detection and classification across four distinct categories.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">CNN</div>
            <div class="hero-stat-label">Architecture</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">4</div>
            <div class="hero-stat-label">Classifications</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">150²</div>
            <div class="hero-stat-label">Input Resolution</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">&lt;2s</div>
            <div class="hero-stat-label">Analysis Time</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# About & Model Info
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="feature-icon">◇</div>
        <div class="feature-title" style="font-size: 1.25rem; margin-bottom: 1rem;">About This Project</div>
        <div class="tumor-desc">
            This application utilizes a Convolutional Neural Network trained on extensive 
            MRI datasets to classify brain tumors into four categories. Developed as an 
            educational resource for medical students, researchers, and healthcare professionals 
            exploring AI-assisted diagnostic methodologies.
        </div>
        <ul class="tumor-list" style="margin-top: 1.5rem;">
            <li>Designed for educational purposes</li>
            <li>Research-grade classification accuracy</li>
            <li>Real-time inference capabilities</li>
            <li>Transparent probability outputs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="feature-icon">◆</div>
        <div class="feature-title" style="font-size: 1.25rem; margin-bottom: 1rem;">Model Specifications</div>
        <ul class="tumor-list">
            <li><strong style="color: #f0fdfa;">Architecture</strong> — Convolutional Neural Network</li>
            <li><strong style="color: #f0fdfa;">Input Dimensions</strong> — 150 × 150 × 3 (RGB)</li>
            <li><strong style="color: #f0fdfa;">Output Layer</strong> — Softmax (4 classes)</li>
            <li><strong style="color: #f0fdfa;">Framework</strong> — TensorFlow / Keras</li>
            <li><strong style="color: #f0fdfa;">Dataset</strong> — Brain Tumor MRI Collection</li>
            <li><strong style="color: #f0fdfa;">Preprocessing</strong> — Normalized [0, 1]</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Classification Categories Section
st.markdown("""
<div class="section-header">
    <div class="section-tag">◎ Classification Categories</div>
    <h2 class="section-title">Tumor Type Reference</h2>
    <p class="section-desc">Understanding the four classification categories used by the neural network</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-glioma">
        <div class="tumor-icon tumor-icon-glioma">G</div>
        <div class="tumor-name">Glioma</div>
        <span class="tumor-severity {TUMOR_INFO['glioma']['severity_class']}">{TUMOR_INFO['glioma']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['glioma']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['glioma']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-pituitary">
        <div class="tumor-icon tumor-icon-pituitary">P</div>
        <div class="tumor-name">Pituitary</div>
        <span class="tumor-severity {TUMOR_INFO['pituitary']['severity_class']}">{TUMOR_INFO['pituitary']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['pituitary']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['pituitary']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-meningioma">
        <div class="tumor-icon tumor-icon-meningioma">M</div>
        <div class="tumor-name">Meningioma</div>
        <span class="tumor-severity {TUMOR_INFO['meningioma']['severity_class']}">{TUMOR_INFO['meningioma']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['meningioma']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['meningioma']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-notumor">
        <div class="tumor-icon tumor-icon-notumor">N</div>
        <div class="tumor-name">No Tumor</div>
        <span class="tumor-severity {TUMOR_INFO['notumor']['severity_class']}">{TUMOR_INFO['notumor']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['notumor']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['notumor']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Divider
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Upload Section
st.markdown("""
<div class="section-header">
    <div class="section-tag">◉ Image Analysis</div>
    <h2 class="section-title">Upload MRI Scan</h2>
    <p class="section-desc">Submit a brain MRI scan for automated classification</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="upload-container">
    <div class="upload-icon">↑</div>
    <div class="upload-title">Drop your MRI scan here</div>
    <div class="upload-subtitle">or click to browse from your device</div>
    <div class="upload-formats">
        <span class="format-badge">PNG</span>
        <span class="format-badge">JPG</span>
        <span class="format-badge">JPEG</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="color: #5eead4; font-weight: 600; margin-bottom: 1rem;">◇ Uploaded Scan</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="color: #5eead4; font-weight: 600; margin-bottom: 1rem;">◆ Classification Results</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶ Run Analysis", type="primary"):
            with st.spinner("Processing scan..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.012)
                    progress_bar.progress(i + 1)
                
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                class_name = CLASS_NAMES[predicted_class]
                
                progress_bar.empty()
                
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-header">
                        <div class="result-label">Primary Classification</div>
                        <div class="result-value">{class_name.upper()}</div>
                        <div class="result-confidence">Confidence: <span>{confidence:.1%}</span></div>
                    </div>
                    <div style="color: #6b7280; font-size: 0.85rem; font-weight: 500; margin-bottom: 1rem;">
                        Probability Distribution
                    </div>
                """, unsafe_allow_html=True)
                
                for cls, score in zip(CLASS_NAMES, prediction[0]):
                    st.markdown(f"""
                    <div class="prob-item">
                        <div class="prob-header">
                            <span class="prob-name">{cls.capitalize()}</span>
                            <span class="prob-value" style="color: {CLASS_COLORS[cls]};">{score:.1%}</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar prob-bar-{cls}" style="width: {score*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-icon">!</div>
    <div class="disclaimer-text">
        <strong>Medical Disclaimer:</strong> This application is developed exclusively for educational and 
        research purposes. It should not be used as a substitute for professional medical diagnosis. 
        Always consult qualified healthcare professionals for clinical decisions.
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-links">
        <span class="footer-link">Documentation</span>
        <span class="footer-link">Research</span>
        <span class="footer-link">GitHub</span>
        <span class="footer-link">Contact</span>
    </div>
    <div class="footer-copyright">
        © 2024 BrainTumor — Built with TensorFlow & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

