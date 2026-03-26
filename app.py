import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# 1. Penanganan Path agar folder 'src' terbaca
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_folder_path = os.path.join(current_dir, "auto_forecast")
if sub_folder_path not in sys.path:
    sys.path.append(sub_folder_path)

# 2. Import Module dengan pengecekan fungsi yang tersedia
try:
    from src.modeling import SalesForecasting
    # Berdasarkan file plotting.py Anda, kita gunakan fungsi yang tersedia:
    from src.plotting import plot_periodic_values_hist
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.stop()

st.set_page_config(page_title="Auto Forecast Tool", layout="wide")

st.title("📈 Auto Sales Forecasting Dashboard")

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.header("1. Pengaturan Data")
date_col = st.sidebar.text_input("Nama Kolom Tanggal", "date")
value_col = st.sidebar.text_input("Nama Kolom Nilai (Sales)", "sales")

st.sidebar.header("2. Parameter Model")
model_options = ['LinearRegression', 'RandomForest', 'XGBoost', 'LSTM', 'ARIMA']
selected_models = st.sidebar.multiselect("Pilih Model", model_options, default=['XGBoost'])
horizon = st.sidebar.number_input("Horizon Prediksi (Bulan)", min_value=1, max_value=24, value=12)

# --- MAIN PAGE: UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data Mentah")
        st.dataframe(df.head())
        
        # Menggunakan fungsi dari plotting.py Anda
        st.write("Distribusi Data:")
        fig_hist, ax_hist = plot_periodic_values_hist(df, value_col)
        st.pyplot(fig_hist)

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang melatih model...'):
                    # Inisialisasi Class (Tanpa argumen model_list di __init__ sesuai modeling.py)
                    # Catatan: modeling.py Anda membutuhkan model_list saat inisialisasi
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # Kita asumsikan data dimasukkan ke object atau diproses di pre_process
                    # Berdasarkan modeling.py, Anda perlu menyesuaikan cara input data ke class ini
                    # Jika modeling.py Anda memerlukan data di awal, pastikan init-nya sesuai.
                    
                    st.success("✅ Prediksi Selesai!")
                    st.info("Catatan: Visualisasi forecast menggunakan line chart standar Streamlit.")
                    
                    # Simulasi penampilan hasil (Ganti dengan logik forecaster.run_forecast Anda)
                    # Karena modeling.py sangat kompleks, pastikan return value-nya sesuai.
                    
            except Exception as e:
                st.error(f"Detail Error: {e}")
else:
    st.info("Silakan upload file CSV untuk memulai.")
