import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
class_mapping = {
    0: 'india gate',
    1: 'gateway of india',
    2: 'golden temple',
    3: 'hawa mahal',
    4: 'charminar',
    5: 'lotus_temple',
    6: 'qutub_minar',
    7: 'sun temple konark',
    8: 'tajmahal',
    9: 'victoria memorial'
}

# Load your trained model
model = tf.keras.models.load_model('my_combined_model (3).h5')

# Function to preprocess the image so it can be fed to the model
def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((300, 300))  # Replace with the size your model expects
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title('Monument Identification App')

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpeg', 'jpg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    # Display the prediction
    st.write(f"The model predicts this monument as: {class_mapping[predicted_class]}")
