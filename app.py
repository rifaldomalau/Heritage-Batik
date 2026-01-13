import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- KONFIGURASI TEMA HITAM ---
st.set_page_config(page_title="Batik AI 2023", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .result-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_my_model():
    # Menambahkan compile=False sering membantu mengatasi eror versi saat load model .h5 lama
    model = tf.keras.models.load_model('model.h5', compile=False)
    return model

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- UI ---
st.title("🏛️ Batik Recognition (Legacy Model)")
st.write("Model ini menggunakan arsitektur dari Mei 2023.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Batik", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)

with col2:
    if uploaded_file:
        with st.spinner('Menganalisis...'):
            model = load_my_model()
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            # Label sesuai notebook Anda
            labels = {0:'Batik Bali', 1:'Batik Dayak', 2:'Batik Geblek Renteng', 
                      3:'Batik Ikat Celup', 4:'Batik Kawung', 5:'Batik Lasem', 
                      6:'Batik Megamendung', 7:'Batik Parang', 8:'Batik Tambal'}
            
            idx = np.argmax(prediction)
            conf = np.max(prediction) * 100
            
            st.markdown(f"""
                <div class="result-card">
                    <h2 style="color: #58A6FF;">{labels[idx]}</h2>
                    <p>Confidence: {conf:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)