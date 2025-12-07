import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .tumor-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .result-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e5e7eb;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()
IMG_SIZE = (150, 150)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

TUMOR_INFO = {
    "glioma": {
        "emoji": "🔴",
        "desc": "Arises from glial cells in the brain/spine. Can be benign or malignant.",
        "severity": "Variable (Grade I-IV)"
    },
    "meningioma": {
        "emoji": "🟠", 
        "desc": "Develops from meninges (brain/spinal cord membranes). Usually benign.",
        "severity": "Typically Benign"
    },
    "pituitary": {
        "emoji": "🟡",
        "desc": "Grows in pituitary gland. Most are benign adenomas.",
        "severity": "Usually Benign"
    },
    "notumor": {
        "emoji": "🟢",
        "desc": "No tumor detected in the MRI scan.",
        "severity": "Normal"
    }
}

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# Header
st.markdown('<h1 class="main-header">🧠 Brain Tumor CNN Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered MRI analysis for educational purposes</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI', use_container_width=True)
        
        if st.button("🔬 Analyze Scan", type="primary", use_container_width=True):
            with st.spinner("Running CNN analysis..."):
                processed = preprocess_image(image)
                prediction = model.predict(processed)
                pred_class = np.argmax(prediction[0])
                confidence = prediction[0][pred_class]
                
                st.session_state.result = {
                    "class": CLASS_NAMES[pred_class],
                    "confidence": confidence,
                    "all_scores": prediction[0]
                }

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if "result" in st.session_state:
        r = st.session_state.result
        info = TUMOR_INFO[r["class"]]
        
        st.markdown(f"""
        <div class="result-success">
            <h2>{info["emoji"]} {r["class"].upper()}</h2>
            <p style="font-size: 1.5rem; margin: 0;">Confidence: <strong>{r["confidence"]:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card">
            <strong>About this classification:</strong><br>
            {info["desc"]}<br><br>
            <strong>Severity:</strong> {info["severity"]}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Probability Distribution")
        for i, score in enumerate(r["all_scores"]):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(float(score), text=f"{TUMOR_INFO[CLASS_NAMES[i]]['emoji']} {CLASS_NAMES[i]}")
            with col_b:
                st.write(f"{score:.1%}")
    else:
        st.info("Upload an MRI scan and click 'Analyze' to see results")

# Educational section
st.markdown("---")
st.markdown("### 📚 Learn About Brain Tumors")
cols = st.columns(4)
for i, (name, info) in enumerate(TUMOR_INFO.items()):
    with cols[i]:
        st.markdown(f"""
        <div class="tumor-card">
            <h4>{info["emoji"]} {name.title()}</h4>
            <p style="font-size: 0.85rem; color: #6b7280;">{info["desc"]}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ For educational purposes only. Not a medical diagnostic tool. Always consult healthcare professionals.")
