import streamlit as st
import pandas as pd
import sys
import os

# Memastikan folder src terbaca sebagai modul
sys.path.append(os.getcwd())

from src.modeling import SalesForecasting
from src.plotting import plot_results # Pastikan fungsi ini ada di plotting.py

st.set_page_config(page_title="Auto Forecast Tool", layout="wide")

st.title("📈 Auto Sales Forecasting Dashboard")
st.markdown("Aplikasi ini menggunakan XGBoost, LSTM, dan model lainnya untuk memprediksi penjualan.")

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.header("1. Pengaturan Data")
date_col = st.sidebar.text_input("Nama Kolom Tanggal", "date")
value_col = st.sidebar.text_input("Nama Kolom Nilai (Sales)", "sales")

st.sidebar.header("2. Parameter Model")
model_options = ['LinearRegression', 'RandomForest', 'XGBoost', 'LSTM', 'ARIMA']
selected_models = st.sidebar.multiselect("Pilih Model", model_options, default=['XGBoost', 'LinearRegression'])
horizon = st.sidebar.number_input("Horizon Prediksi (Bulan)", min_value=1, max_value=24, value=12)

# --- MAIN PAGE: UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data Mentah")
        st.write(df.head())
        st.line_chart(df.set_index(date_col)[value_col])

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang memproses data dan melatih model...'):
                    # 1. Inisialisasi Class
                    forecaster = SalesForecasting(
                        data=df, 
                        date_col=date_col, 
                        value_col=value_col, 
                        model_list=selected_models
                    )
                    
                    # 2. Pre-processing (Penting sesuai isi modeling.py)
                    forecaster.pre_process()
                    
                    # 3. Jalankan Forecast
                    forecast_results = forecaster.run_forecast(forecast_horizon=horizon)
                    
                    st.success("✅ Prediksi Selesai!")
                    
                    # 4. Tampilkan Hasil
                    st.subheader("Hasil Prediksi")
                    st.dataframe(forecast_results)
                    
                    # 5. Visualisasi (Jika plotting.py mendukung)
                    # st.pyplot(plot_results(forecast_results))
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan upload file CSV melalui sidebar untuk memulai.")
