import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time

# Page config
st.set_page_config(
    page_title="BrainTumorAI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html { scroll-behavior: smooth; }
    
    .stApp {
        background: linear-gradient(160deg, #0a0f0f 0%, #0d1b1b 40%, #112020 100%);
        padding: 0 !important;
    }
    
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Navigation Bar */
    .navbar {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2.5rem;
        background: rgba(13, 27, 27, 0.98);
        backdrop-filter: blur(25px);
        border-bottom: 1px solid rgba(94, 234, 212, 0.15);
        width: 100%;
        max-width: 1400px;
        z-index: 1000;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nav-logo-icon {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #14b8a6, #5eead4);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        color: #0a0f0f;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
    
    .nav-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f0fdfa;
        background: linear-gradient(135deg, #f0fdfa, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .nav-links {
        display: flex;
        gap: 1rem;
    }
    
    .nav-link {
        color: #d1d5db !important;
        font-size: 0.95rem;
        font-weight: 500;
        text-decoration: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .nav-link:hover {
        color: #5eead4 !important;
        background: rgba(94, 234, 212, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(94, 234, 212, 0.2);
    }
    
    .section-anchor {
        scroll-margin-top: 120px;
        padding-top: 120px;
    }
    
    .section-padding {
        padding: 5rem 0;
    }
    
    /* Hero Section */
    .hero-content {
        text-align: center;
        max-width: 900px;
        margin: 0 auto;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(94, 234, 212, 0.15);
        border: 1px solid rgba(94, 234, 212, 0.3);
        color: #5eead4;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 2rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(94, 234, 212, 0.1);
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        color: #f0fdfa;
        margin-bottom: 1.5rem;
        line-height: 1.1;
    }
    
    .hero-title span {
        background: linear-gradient(135deg, #14b8a6 0%, #5eead4 50%, #99f6e4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        color: #a1a1aa;
        font-size: 1.2rem;
        font-weight: 300;
        max-width: 700px;
        margin: 0 auto 3rem;
        line-height: 1.75;
        text-align: center;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 3.5rem;
        padding: 2.5rem;
        background: rgba(20, 184, 166, 0.05);
        border-radius: 20px;
        border: 1px solid rgba(94, 234, 212, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .hero-stat {
        text-align: center;
    }
    
    .hero-stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 800;
        background: linear-gradient(135deg, #5eead4, #99f6e4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .hero-stat-label {
        color: #9ca3af;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Section Headers - CENTERED */
    .section-header {
        text-align: center;
        margin-bottom: 4rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .section-tag {
        color: #14b8a6;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: block;
        text-align: center;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.75rem, 4vw, 2.5rem);
        font-weight: 700;
        color: #f0fdfa;
        margin-bottom: 1rem;
        line-height: 1.2;
        text-align: center;
    }
    
    .section-desc {
        color: #9ca3af;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.7;
        text-align: center;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(20, 30, 30, 0.6);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(94, 234, 212, 0.12);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        border-color: rgba(94, 234, 212, 0.25);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
        background: rgba(20, 30, 30, 0.75);
    }
    
    /* Tumor Cards */
    .tumor-card {
        position: relative;
        overflow: hidden;
        min-height: 380px;
    }
    
    .tumor-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        border-radius: 4px 0 0 4px;
    }
    
    .tumor-card-glioma::before { background: linear-gradient(180deg, #f43f5e, #e11d48); }
    .tumor-card-meningioma::before { background: linear-gradient(180deg, #f59e0b, #d97706); }
    .tumor-card-pituitary::before { background: linear-gradient(180deg, #a855f7, #7c3aed); }
    .tumor-card-notumor::before { background: linear-gradient(180deg, #10b981, #059669); }
    
    .tumor-icon {
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    .tumor-icon-glioma { background: rgba(244, 63, 94, 0.2); color: #f43f5e; }
    .tumor-icon-meningioma { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .tumor-icon-pituitary { background: rgba(168, 85, 247, 0.2); color: #a855f7; }
    .tumor-icon-notumor { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    
    .tumor-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0fdfa;
        margin-bottom: 0.75rem;
    }
    
    .tumor-severity {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding: 0.35rem 1rem;
        border-radius: 25px;
        display: inline-block;
    }
    
    .severity-high { background: rgba(244, 63, 94, 0.15); color: #f43f5e; }
    .severity-moderate { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
    .severity-variable { background: rgba(168, 85, 247, 0.15); color: #a855f7; }
    .severity-none { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    
    .tumor-desc {
        color: #d1d5db;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1.75rem;
        flex-grow: 1;
    }
    
    .tumor-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tumor-list li {
        color: #9ca3af;
        font-size: 0.9rem;
        padding: 0.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .tumor-list li::before {
        content: '→';
        color: #5eead4;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.2rem;
        flex-shrink: 0;
    }
    
    /* Info Cards */
    .info-card {
        min-height: 320px;
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        background: rgba(94, 234, 212, 0.15);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #5eead4;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    .feature-title {
        color: #f0fdfa;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1.25rem;
    }
    
    /* Upload Section */
    .upload-zone {
        max-width: 800px;
        margin: 0 auto 3rem;
    }
    
    .upload-container {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.08), rgba(94, 234, 212, 0.03));
        border: 2px dashed rgba(94, 234, 212, 0.25);
        border-radius: 20px;
        padding: 4rem 3rem;
        text-align: center;
        transition: all 0.4s ease;
        margin: 2rem 0;
    }
    
    .upload-container:hover {
        border-color: rgba(94, 234, 212, 0.5);
        box-shadow: 0 0 50px rgba(94, 234, 212, 0.15);
        transform: translateY(-4px);
    }
    
    .upload-icon {
        width: 72px;
        height: 72px;
        background: rgba(94, 234, 212, 0.2);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.75rem;
        font-size: 1.8rem;
        color: #5eead4;
        box-shadow: 0 8px 25px rgba(94, 234, 212, 0.2);
    }
    
    .upload-title {
        color: #f0fdfa;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .upload-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .upload-formats {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .format-badge {
        background: rgba(94, 234, 212, 0.15);
        color: #5eead4;
        padding: 0.5rem 1.25rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Results Section */
    .results-container {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .result-main {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .result-main-icon {
        width: 80px;
        height: 80px;
        background: rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        font-size: 2.5rem;
        color: #10b981;
    }
    
    .result-main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
        margin-bottom: 0.75rem;
    }
    
    .result-main-confidence {
        font-size: 1.3rem;
        color: #059669;
        font-weight: 700;
    }
    
    /* Probability Bars */
    .prob-container {
        background: rgba(20, 30, 30, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
    }
    
    .prob-item {
        margin-bottom: 1.5rem;
    }
    
    .prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .prob-name {
        color: #e5e7eb;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .prob-value {
        font-weight: 700;
        font-size: 1rem;
    }
    
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    
    .prob-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .prob-bar-glioma { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .prob-bar-meningioma { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .prob-bar-pituitary { background: linear-gradient(90deg, #a855f7, #c084fc); }
    .prob-bar-notumor { background: linear-gradient(90deg, #10b981, #34d399); }
    
    /* Disclaimer */
    .disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 4rem auto 4rem;
        max-width: 900px;
        display: flex;
        align-items: flex-start;
        gap: 1.25rem;
    }
    
    .disclaimer-icon {
        width: 28px;
        height: 28px;
        background: rgba(245, 158, 11, 0.25);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f59e0b;
        font-weight: 800;
        font-size: 1rem;
        flex-shrink: 0;
        margin-top: 0.2rem;
    }
    
    .disclaimer-text {
        color: #d1d5db;
        font-size: 1rem;
        line-height: 1.7;
        flex: 1;
    }
    
    .disclaimer-text strong {
        color: #f59e0b;
        font-weight: 700;
    }
    
    /* Footer */
    .footer {
        padding: 4rem 0 2rem;
        margin-top: 6rem;
        border-top: 1px solid rgba(94, 234, 212, 0.1);
        text-align: center;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: #9ca3af;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    
    .footer-link:hover {
        color: #5eead4;
        background: rgba(94, 234, 212, 0.1);
        transform: translateY(-1px);
    }
    
    .footer-copyright {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(94, 234, 212, 0.3), transparent);
        margin: 5rem auto;
        width: 200px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        color: #0a0f0f !important;
        border: none;
        padding: 1.1rem 3rem;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        max-width: 300px;
        box-shadow: 0 8px 25px rgba(20, 184, 166, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 40px rgba(20, 184, 166, 0.4);
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
    }
    
    /* Hide Streamlit defaults */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    .stImage > img {
        border-radius: 16px !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3) !important;
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
        "icon": "✓",
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

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
<nav class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-icon">◉</div>
        <div class="nav-title">BrainTumorAI</div>
    </div>
    <div class="nav-links">
        <a href="#overview" class="nav-link">Overview</a>
        <a href="#documentation" class="nav-link">Documentation</a>
        <a href="#categories" class="nav-link">Categories</a>
        <a href="#analysis" class="nav-link">Analyze</a>
    </div>
</nav>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div id="overview" class="section-anchor"></div>
<div class="section-padding">
    <div class="hero-content">
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
                <div class="hero-stat-label">Classes</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">150²</div>
                <div class="hero-stat-label">Resolution</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">&lt;2s</div>
                <div class="hero-stat-label">Inference</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Documentation Section
st.markdown("""
<div id="documentation" class="section-anchor"></div>
<div class="section-padding">
    <div class="section-header">
        <div class="section-tag">Technical Overview</div>
        <h2 class="section-title">Project Documentation</h2>
        <p class="section-desc">Technical specifications and model architecture details</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"""
    <div class="glass-card info-card">
        <div class="feature-icon">◇</div>
        <div class="feature-title">About This Project</div>
        <p class="tumor-desc">
            Educational AI application for brain tumor classification using 
            Convolutional Neural Networks trained on extensive MRI datasets.
        </p>
        <ul class="tumor-list">
            <li>Developed for medical education</li>
            <li>Research-grade accuracy</li>
            <li>Real-time inference</li>
            <li>Transparent predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-card info-card">
        <div class="feature-icon">◆</div>
        <div class="feature-title">Model Architecture</div>
        <ul class="tumor-list">
            <li><strong>CNN Layers:</strong> Multiple Conv2D + MaxPooling</li>
            <li><strong>Input:</strong> 150×150×3 RGB MRI images</li>
            <li><strong>Output:</strong> Softmax (4 classes)</li>
            <li><strong>Framework:</strong> TensorFlow/Keras 2.x</li>
            <li><strong>Preprocessing:</strong> Rescale [0,1] normalization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Categories Section
st.markdown('<div id="categories" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-padding">
    <div class="section-header">
        <div class="section-tag">Classification Categories</div>
        <h2 class="section-title">Tumor Type Reference</h2>
        <p class="section-desc">Detailed overview of the four brain tumor classifications</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-glioma">
        <div class="tumor-icon tumor-icon-glioma">{TUMOR_INFO['glioma']['icon']}</div>
        <div class="tumor-name">Glioma</div>
        <span class="tumor-severity {TUMOR_INFO['glioma']['severity_class']}">{TUMOR_INFO['glioma']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['glioma']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['glioma']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-meningioma">
        <div class="tumor-icon tumor-icon-meningioma">{TUMOR_INFO['meningioma']['icon']}</div>
        <div class="tumor-name">Meningioma</div>
        <span class="tumor-severity {TUMOR_INFO['meningioma']['severity_class']}">{TUMOR_INFO['meningioma']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['meningioma']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['meningioma']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-pituitary">
        <div class="tumor-icon tumor-icon-pituitary">{TUMOR_INFO['pituitary']['icon']}</div>
        <div class="tumor-name">Pituitary</div>
        <span class="tumor-severity {TUMOR_INFO['pituitary']['severity_class']}">{TUMOR_INFO['pituitary']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['pituitary']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['pituitary']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="glass-card tumor-card tumor-card-notumor">
        <div class="tumor-icon tumor-icon-notumor">{TUMOR_INFO['notumor']['icon']}</div>
        <div class="tumor-name">No Tumor</div>
        <span class="tumor-severity {TUMOR_INFO['notumor']['severity_class']}">{TUMOR_INFO['notumor']['severity']}</span>
        <p class="tumor-desc">{TUMOR_INFO['notumor']['description']}</p>
        <ul class="tumor-list">
            {''.join([f'<li>{point}</li>' for point in TUMOR_INFO['notumor']['points']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Analysis Section
st.markdown('<div id="analysis" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-padding">
    <div class="section-header">
        <div class="section-tag">MRI Analysis</div>
        <h2 class="section-title">Upload & Analyze</h2>
        <p class="section-desc">Submit brain MRI scans for instant classification</p>
    </div>
    
    <div class="upload-zone">
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload MRI Scan", 
    type=["png", "jpg", "jpeg"], 
    label_visibility="collapsed",
    help="Supported formats: PNG, JPG, JPEG (Max 10MB)"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <div style="color: #5eead4; font-weight: 700; font-size: 1.1rem; margin-bottom: 1.5rem;">
                ◎ Uploaded MRI Scan
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True, clamp=True)
    
    with col2:
        st.markdown("""
        <div style="color: #5eead4; font-weight: 700; font-size: 1.1rem; margin-bottom: 1.5rem; text-align: center;">
            ◉ Classification Results
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("◆ Run Neural Network Analysis", type="primary"):
            with st.spinner("Analyzing MRI scan..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    if i < 25:
                        status_text.text("→ Preprocessing image...")
                    elif i < 50:
                        status_text.text("→ Loading neural network...")
                    elif i < 80:
                        status_text.text("→ Running inference...")
                    else:
                        status_text.text("→ Generating results...")
                    
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                status_text.empty()
                progress_bar.empty()
                
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class_idx]
                class_name = CLASS_NAMES[predicted_class_idx]
                
                st.markdown(f"""
                <div class="result-main">
                    <div class="result-main-icon">◉</div>
                    <div class="result-main-title">{class_name.upper()}</div>
                    <div class="result-main-confidence">
                        Confidence: <strong>{confidence:.1%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="prob-container">
                    <div style="color: #9ca3af; font-size: 0.9rem; font-weight: 600; margin-bottom: 1.5rem; text-align: center;">
                        ◇ Probability Distribution
                    </div>
                """, unsafe_allow_html=True)
                
                for cls, score in zip(CLASS_NAMES, prediction[0]):
                    color = CLASS_COLORS[cls]
                    st.markdown(f"""
                    <div class="prob-item">
                        <div class="prob-header">
                            <span class="prob-name">{cls.capitalize()}</span>
                            <span class="prob-value" style="color: {color};">{score:.1%}</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar prob-bar-{cls}" style="width: {score*100:.0f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-container">
        <div class="upload-icon">◎</div>
        <div class="upload-title">Drop your MRI scan here</div>
        <div class="upload-subtitle">or click to browse (PNG, JPG, JPEG)</div>
        <div class="upload-formats">
            <span class="format-badge">PNG</span>
            <span class="format-badge">JPG</span>
            <span class="format-badge">JPEG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-icon">!</div>
    <div class="disclaimer-text">
        <strong>Medical Disclaimer:</strong> This AI application is developed exclusively for 
        <strong>educational and research purposes only</strong>. It should <strong>never</strong> 
        be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making. 
        Always consult qualified healthcare professionals for medical advice.
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-links">
        <span class="footer-link">Documentation</span>
        <span class="footer-link">Research Paper</span>
        <span class="footer-link">GitHub</span>
        <span class="footer-link">Contact</span>
    </div>
    <div class="footer-copyright">
        © 2025 BrainTumorAI — Built with TensorFlow, Keras & Streamlit
    </div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height: 120px;"></div>', unsafe_allow_html=True)
