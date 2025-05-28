# app_streamlit.py
import streamlit as st
import cv2
import numpy as np
import joblib

# Load model dan definisikan kategori
model = joblib.load('bismillahpcdtubesfixlagi.pkl')
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']  # Sesuaikan urutan kategori training

def extract_color_histogram(image, bins=(8,8,8)):
    """Fungsi ekstraksi fitur sama seperti saat training"""
    image = (image * 255).astype("uint8")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Interface Streamlit
st.title('🌍 Klasifikasi Citra Satelit')
st.write("""
Aplikasi ini mengklasifikasikan citra satelit ke dalam 4 kategori:
- Cloudy
- Desert
- Green Area
- Water
""")

uploaded_file = st.file_uploader("Unggah gambar satelit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca dan praproses gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Praproses sama seperti data training
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    
    # Ekstraksi fitur
    features = extract_color_histogram(img_normalized)
    
    # Prediksi
    proba = model.predict_proba([features])[0]
    pred_class = model.predict([features])[0]
    
    # Tampilkan hasil
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption='Citra yang Diunggah', use_container_width=True)

    with col2:
        st.subheader("🔄 Hasil Prediksi")
        st.metric(label="Kategori Prediksi", value=CLASS_NAMES[pred_class])
        
        st.subheader("📊 Probabilitas Klasifikasi")
        for class_name, prob in zip(CLASS_NAMES, proba):
            st.progress(prob, text=f"{class_name}: {prob*100:.2f}%")

st.markdown("---")
st.info("⚠️ Pastikan gambar yang diunggah adalah citra satelit dengan format JPG/PNG")