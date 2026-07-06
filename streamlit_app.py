import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Konfigurasi Halaman
st.set_page_config(page_title="WBC Detection & Counting Dashboard", layout="wide")

# 2. Muat Model YOLO (Cached)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 3. Inisialisasi Session State
if 'history' not in st.session_state:
    st.session_state['history'] = []
    
if 'aggregate_counts' not in st.session_state:
    st.session_state['aggregate_counts'] = {
        'Neutrophil': 0, 'Lymphocyte': 0, 'Monocyte': 0, 'Eosinophil': 0, 'Basophil': 0
    }

# State temporary untuk menampung hasil edit sebelum di-submit ke akumulasi
if 'temp_output' not in st.session_state:
    st.session_state['temp_output'] = None

# 4. Pengaturan Sidebar (Menu Samping)
st.sidebar.title("Navigasi & Parameter")
page = st.sidebar.radio("Pilih Halaman:", ["Deteksi Sel", "Riwayat Pemrosesan"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Fitur 1: Parameter Model")

# Slider untuk Confidence Threshold
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.00, max_value=1.00, value=0.25, step=0.05,
    help="Batas minimum keyakinan model untuk mendeteksi objek."
)

# Slider untuk IoU Threshold (NMS)
iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)", 
    min_value=0.00, max_value=1.00, value=0.45, step=0.05,
    help="Mengatur tumpang tindih bounding box (Intersection over Union)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Mode Hitung Jenis")
count_mode = st.sidebar.radio(
    "Tampilkan Grafik & Tabel Berdasarkan:",
    ["Hanya Gambar Saat Ini", "Akumulasi Keseluruhan Sesi"]
)

# Tombol Reset Akumulasi
st.sidebar.markdown("---")
if st.sidebar.button("⚠️ Reset Seluruh Akumulasi", use_container_width=True):
    st.session_state['aggregate_counts'] = {k: 0 for k in st.session_state['aggregate_counts']}
    st.session_state['history'] = []
    st.session_state['temp_output'] = None
    st.sidebar.success("Semua data akumulasi dan riwayat berhasil di-reset!")
    st.rerun()


# 5. Halaman Utama: Deteksi Sel
if page == "Deteksi Sel":
    st.title("Sistem Deteksi dan Klasifikasi Sel Darah Putih")
    st.write("Silakan unggah citra mikroskop apusan darah untuk memulai analisis.")

    uploaded_file = st.file_uploader("Pilih file citra (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # TAMPILAN AWAL (Sebelum Deteksi Berjalan)
        if st.session_state['temp_output'] is None or st.session_state['temp_output']['filename'] != uploaded_file.name:
            col_img, col_btn = st.columns([3, 1])
            with col_img:
                st.image(image, caption="Citra Asli yang Diunggah", use_container_width=True)
            with col_btn:
                st.write("Aksi:")
                if st.button("Jalankan Prediksi YOLO", key="btn_run", use_container_width=True):
                    with st.spinner("Model sedang memproses gambar..."):
                        # Jalankan Model dengan parameter dari Slider
                        results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
                        result = results[0]

                        # Plot Bounding Box
                        img_bgr = result.plot() 
                        img_with_boxes = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                        # Kalkulasi awal dari YOLO
                        current_counts = {k: 0 for k in st.session_state['aggregate_counts']}
                        crops = []
                        
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf_score = float(box.conf[0])
                            class_name = model.names[cls_id]
                            
                            if class_name in current_counts:
                                current_counts[class_name] += 1

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cropped_cell = image.crop((x1, y1, x2, y2))
                            crops.append({
                                "image": cropped_cell,
                                "class": class_name,
                                "conf": conf_score
                            })

                        # Simpan hasil sementara ke session state (BELUM masuk riwayat permanen)
                        st.session_state['temp_output'] = {
                            'filename': uploaded_file.name,
                            'img_visual': img_with_boxes,
                            'counts': current_counts,
                            'crops': crops
                        }
                    st.rerun()

        # TAMPILAN SETELAH DETEKSI (Proses Edit & Konfirmasi)
        if st.session_state['temp_output'] is not None and st.session_state['temp_output']['filename'] == uploaded_file.name:
            temp_data = st.session_state['temp_output']
            
            st.success("Deteksi objek awal selesai! Harap tinjau dan sesuaikan jumlah jika diperlukan.")
            
            # ---------------- MENTOR EDIT OUTPUT ----------------
            st.markdown("### ✏️ Form Koreksi Hasil Deteksi")
            st.info("Jika model salah menghitung atau melewatkan sel, ubah angka di bawah ini sebelum menekan tombol simpan.")
            
            # Membuat input numerik yang dinamis berdasarkan hasil YOLO
            edited_counts = {}
            col_edit = st.columns(5)
            for idx, (cell_name, original_count) in enumerate(temp_data['counts'].items()):
                with col_edit[idx]:
                    edited_counts[cell_name] = st.number_input(
                        f"Jumlah {cell_name}", 
                        min_value=0, 
                        value=original_count, 
                        step=1, 
                        key=f"input_{cell_name}"
                    )
            
            col_save1, col_save2 = st.columns([1, 3])
            with col_save1:
                submit_button = st.button("💾 Simpan & Perbarui Total", type="primary", use_container_width=True)
            
            if submit_button:
                # Masukkan hasil yang sudah DIEDIT oleh user ke history dan akumulasi riwayat
                st.session_state['history'].append({
                    'filename': temp_data['filename'],
                    'counts': edited_counts
                })
                for cell, count in edited_counts.items():
                    st.session_state['aggregate_counts'][cell] += count
                
                st.toast("Data berhasil disimpan ke akumulasi sesi!", icon="✅")
                # Hapus state sementara agar bersih kembali
                st.session_state['temp_output'] = None
                st.rerun()
            st.markdown("---")
            # ----------------------------------------------------

            # Bagian Tampilan Visualisasi
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                st.subheader("Visualisasi Lokalisasi")
                st.image(temp_data['img_visual'], caption="Hasil Deteksi YOLOv11", use_container_width=True)
            
            with col_res2:
                st.subheader("Hasil Perhitungan Sementara (YOLO)")
                df_temp = pd.DataFrame(list(temp_data['counts'].items()), columns=['Jenis Sel', 'Jumlah Terdeteksi'])
                st.dataframe(df_temp, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("✂️ Potongan Citra Sel Terdeteksi")
            if len(temp_data['crops']) == 0:
                st.warning("Tidak ada sel darah putih yang terdeteksi.")
            else:
                cols_per_row = 6
                for i in range(0, len(temp_data['crops']), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, crop_data in enumerate(temp_data['crops'][i:i + cols_per_row]):
                        with cols[j]:
                            st.image(crop_data["image"], use_container_width=True)
                            st.caption(f"**{crop_data['class']}**\nConf: {crop_data['conf']:.2f}")

    # ---------------- DASHBOARD ANALISIS & EKSPOR (FITUR 2 & 3) ----------------
    if len(st.session_state['history']) > 0:
        st.markdown("---")
        st.header("📊 Analisis Data Medis & Laporan Resmi")
        
        # Menentukan data mana yang mau dianalisis berdasarkan pilihan sidebar
        if count_mode == "Hanya Gambar Saat Ini":
            last_analysis = st.session_state['history'][-1]
            st.subheader(f"Analisis Statistik: {last_analysis['filename']}")
            df_analytics = pd.DataFrame(list(last_analysis['counts'].items()), columns=['Jenis Sel', 'Jumlah'])
        else:
            st.subheader("Analisis Statistik: Akumulasi Seluruh Sesi Pemeriksaan")
            df_analytics = pd.DataFrame(list(st.session_state['aggregate_counts'].items()), columns=['Jenis Sel', 'Jumlah'])

        col_graph, col_table = st.columns([2, 1])
        
        with col_graph:
            st.markdown("#### **Fitur 2: Grafik Distribusi Sel Jenis (Differential Count)**")
            # Menampilkan Grafik Batang Dinamis Menggunakan Dataframe
            st.bar_chart(data=df_analytics, x='Jenis Sel', y='Jumlah', use_container_width=True)
            
        with col_table:
            st.markdown("#### **Tabel Data Referensi**")
            st.dataframe(df_analytics, use_container_width=True, hide_index=True)
            
            # Fitur 3: Ekspor Laporan Ke CSV
            st.markdown("#### **Fitur 3: Download Dokumen**")
            csv_data = df_analytics.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Laporan (.CSV)",
                data=csv_data,
                file_name=f"Laporan_WBC_{count_mode.replace(' ', '_')}.csv",
                mime='text/csv',
                use_container_width=True,
                help="Klik di sini untuk mengunduh tabel sebaran data laboratorium dalam format CSV."
            )

# 6. Halaman Riwayat Pemrosesan
elif page == "Riwayat Pemrosesan":
    st.title("Riwayat Analisis Citra Resmi")
    st.write("Daftar berkas sampel yang telah dikonfirmasi dan dimasukkan ke data rekam medis.")

    if len(st.session_state['history']) == 0:
        st.info("Belum ada rekam data sampel yang berhasil dikonfirmasi dalam sesi ini.")
    else:
        for idx, item in enumerate(st.session_state['history']):
            with st.expander(f"📋 Sampel Pasien #{idx + 1}: {item['filename']}"):
                st.write("Rincian jumlah sel hasil validasi akhir:")
                df_item = pd.DataFrame(list(item['counts'].items()), columns=['Jenis Sel', 'Jumlah Valid'])
                st.dataframe(df_item, hide_index=True)
