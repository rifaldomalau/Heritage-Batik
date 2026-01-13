import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. SET THEME HITAM (Custom CSS) ---
st.set_page_config(page_title="Batik AI - Dark Mode", layout="wide")

st.markdown("""
    <style>
    /* Background utama hitam */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    /* Membuat card untuk hasil agar menonjol */
    .result-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363D;
        text-align: center;
    }
    /* Warna teks sidebar & uploader */
    .stMarkdown, p, h1, h2, h3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIKA MODEL (Tetap dari kode Anda) ---
INV_LABELS = {
    0: 'Batik Bali', 1: 'Batik Dayak', 2: 'Batik Geblek Renteng', 
    3: 'Batik Ikat Celup', 4: 'Batik Kawung', 5: 'Batik Lasem', 
    6: 'Batik Megamendung', 7: 'Batik Parang', 8: 'Batik Tambal'
}

@st.cache_resource
def load_my_model():
    # Pastikan file model .h5 ada di direktori yang sama
    model = tf.keras.models.load_model('model.h5')
    return model

def preprocess_image(image):
    img = image.resize((224, 224)) 
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

# --- 3. TAMPILAN UI ---
st.title("🏛️ Batik Pattern Recognition")
st.write("Sistem cerdas identifikasi motif batik nusantara.")
st.markdown("---")

# Menggunakan kolom agar layout tidak kaku
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Motif")
    uploaded_file = st.file_uploader("Pilih gambar batik...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Preview Gambar', use_column_width=True)

with col2:
    st.subheader("🎯 Hasil Analisis")
    if uploaded_file is not None:
        with st.spinner('Menganalisis pola...'):
            # Prediksi
            model = load_my_model()
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            
            label_index = np.argmax(prediction, axis=1)[0]
            conf_score = np.max(prediction) * 100
            
            # Tampilan hasil di dalam box dark
            st.markdown(f"""
                <div class="result-card">
                    <p style="margin:0; font-size: 0.9em; color: #8B949E;">MOTIF TERDETEKSI</p>
                    <h2 style="color: #58A6FF; margin: 10px 0;">{INV_LABELS[label_index]}</h2>
                    <p style="margin:0; font-size: 1.1em;">Confidence: <b>{conf_score:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Bar akurasi
            st.progress(int(conf_score))
    else:
        st.info("Silahkan upload gambar untuk melihat prediksi.")

# Footer
st.markdown("<br><hr><center style='opacity:0.5;'>©2023 Sudah Milik INTEL. All rights reserved.</center>", unsafe_allow_html=True)