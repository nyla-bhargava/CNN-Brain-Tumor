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

# Preprocess image (change 224,224 to your model's input size)
IMG_SIZE = (224, 224)  # EDIT THIS if different
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🧠 CNN Model Demo")
st.write("Upload an image to test your model!")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("🔮 Predict", type="primary"):
        with st.spinner("Predicting..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            st.success(f"**Prediction:** {CLASS_NAMES[predicted_class]}")
            st.info(f"**Confidence:** {confidence:.2%}")
            st.write("**All predictions:**")
            for i, score in enumerate(prediction[0]):
                st.write(f"{CLASS_NAMES[i]}: {score:.2%}")
