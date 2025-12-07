import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your model (cached so it loads once)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# CORRECTED: Your model's exact input size + 4 classes
IMG_SIZE = (150, 150)  
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Handle grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    
    # Ensure correct shape: (1, 150, 150, 3)
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)

# Streamlit UI
st.title("🧠 Brain Tumor CNN Classifier")
st.write("Upload MRI image to classify: glioma, meningioma, pituitary, or no tumor")

uploaded_file = st.file_uploader("Choose MRI...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', use_column_width=True)
    
    if st.button("🔮 Predict Tumor", type="primary"):
        with st.spinner("Classifying..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            st.success(f"**Predicted:** {CLASS_NAMES[predicted_class]}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
            st.write("**All predictions:**")
            for i, score in enumerate(prediction[0]):
                st.write(f"{CLASS_NAMES[i]}: {score:.2%}")

