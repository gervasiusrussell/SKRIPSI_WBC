import streamlit as st
import pandas as pd
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(page_title="WBC Detection & Counting", layout="wide")

# 2. Inisialisasi Session State (Memori Penyimpanan Sementara)
# Menyimpan riwayat gambar yang sudah diproses
if 'history' not in st.session_state:
    st.session_state['history'] = []
    
# Menyimpan jumlah sel yang diakumulasikan
if 'aggregate_counts' not in st.session_state:
    st.session_state['aggregate_counts'] = {
        'Neutrophil': 0, 
        'Lymphocyte': 0, 
        'Monocyte': 0, 
        'Eosinophil': 0, 
        'Basophil': 0
    }

# 3. Pengaturan Sidebar (Menu Samping)
st.sidebar.title("Navigasi & Pengaturan")
page = st.sidebar.radio("Pilih Halaman:", ["Deteksi Sel", "Riwayat Pemrosesan"])

st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan Perhitungan")
count_mode = st.sidebar.radio(
    "Mode Perhitungan:",
    ["Hanya Gambar Saat Ini", "Akumulasi (Aggregate)"]
)

# 4. Halaman Utama: Deteksi Sel
if page == "Deteksi Sel":
    st.title("Sistem Deteksi dan Klasifikasi Sel Darah Putih")
    st.write("Silakan unggah citra apusan darah tepi untuk memulai deteksi objek.")

    # Fitur Unggah File
    uploaded_file = st.file_uploader("Pilih file citra (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Membuka dan menampilkan gambar asli
        image = Image.open(uploaded_file)
        
        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.image(image, caption="Citra Asli yang Diunggah", use_column_width=True)
            
        with col_btn:
            st.write("Aksi:")
            run_button = st.button("Jalankan Deteksi")

        # Jika tombol deteksi ditekan
        if run_button:
            with st.spinner("Model sedang memproses citra..."):
                # SIMULASI HASIL MODEL:
                # Bagian ini nantinya akan diganti dengan pemanggilan model YOLO kamu
                # Hasil di bawah ini hanya data palsu (dummy)
                dummy_current_counts = {
                    'Neutrophil': 4,
                    'Lymphocyte': 2,
                    'Monocyte': 1,
                    'Eosinophil': 0,
                    'Basophil': 0
                }

                # Simpan hasil ke dalam riwayat (history)
                st.session_state['history'].append({
                    'filename': uploaded_file.name,
                    'counts': dummy_current_counts
                })

                # Tambahkan jumlah sel ke mode akumulasi (aggregate)
                for cell, count in dummy_current_counts.items():
                    st.session_state['aggregate_counts'][cell] += count

            st.success("Proses deteksi selesai!")

            # Menampilkan Hasil Visualisasi dan Tabel
            st.markdown("### Hasil Analisis")
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                st.subheader("Visualisasi Bounding Box")
                # Sementara menampilkan gambar asli, nanti diganti dengan output gambar dari model YOLO
                st.image(image, caption="Hasil Deteksi (Simulasi)", use_column_width=True)

            with col_result2:
                st.subheader("Laporan Hitung Jenis (Differential Count)")

                # Logika peralihan mode perhitungan
                if count_mode == "Hanya Gambar Saat Ini":
                    st.info("Mode: Menampilkan jumlah sel pada citra ini saja.")
                    df_counts = pd.DataFrame(list(dummy_current_counts.items()), columns=['Jenis Sel', 'Jumlah'])
                else:
                    st.info("Mode: Menampilkan total akumulasi dari seluruh citra yang pernah diunggah.")
                    df_counts = pd.DataFrame(list(st.session_state['aggregate_counts'].items()), columns=['Jenis Sel', 'Jumlah Total'])

                # Tampilkan sebagai tabel interaktif
                st.dataframe(df_counts, use_container_width=True)

# 5. Halaman Riwayat Pemrosesan
elif page == "Riwayat Pemrosesan":
    st.title("Riwayat Analisis Citra")

    if len(st.session_state['history']) == 0:
        st.info("Belum ada citra yang diproses dalam sesi ini.")
    else:
        # Menampilkan daftar gambar yang sudah diproses menggunakan expander
        for idx, item in enumerate(st.session_state['history']):
            with st.expander(f"Analisis #{idx + 1}: {item['filename']}"):
                st.write("Rincian sel yang ditemukan:")
                st.json(item['counts'])

        st.markdown("---")
        # Tombol untuk mereset semua data
        if st.button("Hapus Riwayat & Reset Akumulasi"):
            st.session_state['history'] = []
            st.session_state['aggregate_counts'] = {k: 0 for k in st.session_state['aggregate_counts']}
            st.success("Seluruh riwayat dan akumulasi berhasil dihapus!")
            st.rerun()
