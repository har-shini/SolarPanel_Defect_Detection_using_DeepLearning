
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("solar_model.h5")  
    return model

model = load_model()


CLASS_NAMES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage',
               'Physical-Damage', 'Snow-Covered']

IMG_SIZE = (224, 224)


def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0                  
    return img_array


st.set_page_config(page_title="SolarGuard - Solar Panel Defect Classifier", layout="centered")

st.title("Solar Panel Defect Detection")
st.write("Upload an image of a solar panel to detect its condition.")

uploaded_file = st.file_uploader("Upload a Solar Panel Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    
    st.subheader("Classification Result:")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    
    st.subheader("Prediction Probabilities")
    probs = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(probs)
