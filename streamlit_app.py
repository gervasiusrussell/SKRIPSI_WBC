import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Konfigurasi Halaman
st.set_page_config(page_title="WBC Detection & Counting", layout="wide")

# 2. Muat Model YOLO (Gunakan cache agar tidak reload berulang kali)
@st.cache_resource
def load_model():
    # Pastikan file best.pt berada di direktori yang sama, atau sesuaikan path-nya
    return YOLO("best.pt")

model = load_model()

# 3. Inisialisasi Session State (Memori Penyimpanan)
if 'history' not in st.session_state:
    st.session_state['history'] = []
    
if 'aggregate_counts' not in st.session_state:
    st.session_state['aggregate_counts'] = {
        'Neutrophil': 0, 
        'Lymphocyte': 0, 
        'Monocyte': 0, 
        'Eosinophil': 0, 
        'Basophil': 0
    }

# 4. Pengaturan Sidebar
st.sidebar.title("Navigasi & Pengaturan")
page = st.sidebar.radio("Pilih Halaman:", ["Deteksi Sel", "Riwayat Pemrosesan"])

st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan Perhitungan")
count_mode = st.sidebar.radio(
    "Mode Tampilan Laporan:",
    ["Hanya Gambar Saat Ini", "Akumulasi (Aggregate)"]
)

# Tombol Cepat Reset Akumulasi di Sidebar
if st.sidebar.button("⚠️ Reset Total Akumulasi"):
    st.session_state['aggregate_counts'] = {k: 0 for k in st.session_state['aggregate_counts']}
    st.session_state['history'] = []
    st.sidebar.success("Akumulasi di-reset!")
    st.rerun()

# 5. Halaman Utama: Deteksi Sel
if page == "Deteksi Sel":
    st.title("Sistem Deteksi dan Klasifikasi Sel Darah Putih")
    st.write("Silakan unggah citra apusan darah tepi untuk memulai analisis.")

    uploaded_file = st.file_uploader("Pilih file citra (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Buka gambar menggunakan PIL
        image = Image.open(uploaded_file).convert("RGB")
        
        main_container = st.empty()

        # TAMPILAN AWAL (Sebelum Deteksi)
        with main_container.container():
            col_img, col_btn = st.columns([3, 1])
            with col_img:
                st.image(image, caption="Citra Asli yang Diunggah", use_container_width=True)
            with col_btn:
                st.write("Aksi:")
                run_button = st.button("Jalankan Prediksi YOLO", key="btn_run", use_container_width=True)

        # PROSES DETEKSI
        if run_button:
            with st.spinner("Model sedang menganalisis morfologi sel..."):
                # A. Jalankan Inference YOLO
                results = model.predict(image, conf=0.25) # Bisa atur threshold confidence di sini
                result = results[0] # Ambil hasil dari gambar pertama (karena input cuma 1)

                # B. Siapkan Gambar Full dengan Bounding Box
                # result.plot() mengembalikan array BGR dari OpenCV
                img_bgr = result.plot() 
                img_with_boxes = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # C. Ekstrak Data Deteksi & Cropping
                current_counts = {k: 0 for k in st.session_state['aggregate_counts']}
                crops = []
                
                # Looping setiap kotak yang ditemukan YOLO
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    class_name = model.names[cls_id] # Ambil nama kelas asli dari model
                    
                    # Tambah ke hitungan (antisipasi jika nama kelas model sedikit berbeda)
                    if class_name in current_counts:
                        current_counts[class_name] += 1
                    else:
                        current_counts[class_name] = 1

                    # Dapatkan koordinat untuk cropping (x_min, y_min, x_max, y_max)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop dari gambar PIL asli (bukan gambar yang sudah ada bounding box-nya)
                    cropped_cell = image.crop((x1, y1, x2, y2))
                    crops.append({
                        "image": cropped_cell,
                        "class": class_name,
                        "conf": conf_score
                    })

                # D. Simpan ke Session State
                st.session_state['history'].append({
                    'filename': uploaded_file.name,
                    'counts': current_counts
                })

                for cell, count in current_counts.items():
                    if cell in st.session_state['aggregate_counts']:
                        st.session_state['aggregate_counts'][cell] += count

            # TAMPILAN HASIL (Setelah Deteksi)
            main_container.empty() 
            
            with main_container.container():
                st.success(f"Deteksi selesai! Ditemukan {len(crops)} sel.")
                
                # BAGIAN 1: Gambar Full vs Laporan
                col_result1, col_result2 = st.columns([2, 1])

                with col_result1:
                    st.subheader("Visualisasi Keseluruhan")
                    st.image(img_with_boxes, caption="Hasil YOLOv11 (Bounding Box & Class)", use_container_width=True)

                with col_result2:
                    st.subheader("Laporan Hitung Jenis")
                    if count_mode == "Hanya Gambar Saat Ini":
                        st.info("Mode: Single Image")
                        df_counts = pd.DataFrame(list(current_counts.items()), columns=['Jenis Sel', 'Jumlah'])
                    else:
                        st.info("Mode: Total Akumulasi Sesi")
                        df_counts = pd.DataFrame(list(st.session_state['aggregate_counts'].items()), columns=['Jenis Sel', 'Jumlah Total'])

                    st.dataframe(df_counts, use_container_width=True)
                    
                    if st.button("Analisis Gambar Lain", use_container_width=True):
                        st.rerun()

                st.markdown("---")
                
                # BAGIAN 2: Gallery Potongan Sel (Cropped Images)
                st.subheader("Detail Sel Terdeteksi")
                if len(crops) == 0:
                    st.warning("Tidak ada sel darah putih yang terdeteksi pada citra ini.")
                else:
                    # Tampilkan gambar dalam bentuk grid (misal: 6 kolom)
                    cols_per_row = 6
                    for i in range(0, len(crops), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, crop_data in enumerate(crops[i:i + cols_per_row]):
                            with cols[j]:
                                st.image(crop_data["image"], use_container_width=True)
                                # Tampilkan nama dan confidence score di bawah potongan gambar
                                st.caption(f"**{crop_data['class']}**\nConf: {crop_data['conf']:.2f}")

# 6. Halaman Riwayat Pemrosesan
elif page == "Riwayat Pemrosesan":
    st.title("Riwayat Analisis Citra")

    if len(st.session_state['history']) == 0:
        st.info("Belum ada citra yang diproses dalam sesi ini.")
    else:
        for idx, item in enumerate(st.session_state['history']):
            with st.expander(f"Analisis #{idx + 1}: {item['filename']}"):
                st.json(item['counts'])
