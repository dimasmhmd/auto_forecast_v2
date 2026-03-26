import streamlit as st
import pandas as pd
import sys
import os

# 1. Penanganan Path agar folder 'src' terbaca
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_folder_path = os.path.join(current_dir, "auto_forecast")
if sub_folder_path not in sys.path:
    sys.path.append(sub_folder_path)

# 2. Import Module dari folder src
try:
    from src.modeling import SalesForecasting
    from src.plotting import plot_forecast  # Nama fungsi yang benar di plotting.py
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.stop()

st.set_page_config(page_title="Auto Forecast Tool", layout="wide")

st.title("📈 Auto Sales Forecasting Dashboard")
st.markdown("Aplikasi prediksi penjualan menggunakan berbagai model Machine Learning.")

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
        st.dataframe(df.head())
        # Visualisasi data historis sederhana
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
        st.line_chart(df_plot.set_index(date_col)[value_col])

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang memproses data dan melatih model...'):
                    # Inisialisasi dan Jalankan Model
                    forecaster = SalesForecasting(
                        data=df, 
                        date_col=date_col, 
                        value_col=value_col, 
                        model_list=selected_models
                    )
                    
                    forecaster.pre_process()
                    # Hasilnya adalah dictionary {model_name: dataframe_forecast}
                    forecast_dict = forecaster.run_forecast(forecast_horizon=horizon)
                    
                    st.success("✅ Prediksi Selesai!")
                    
                    # Iterasi hasil untuk setiap model yang dipilih
                    for model_name, forecast_df in forecast_dict.items():
                        st.subheader(f"Model: {model_name}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("Tabel Prediksi:")
                            st.dataframe(forecast_df)
                        
                        with col2:
                            st.write("Grafik Forecast:")
                            # Memanggil fungsi plot_forecast dari plotting.py
                            # Fungsi ini memerlukan (original_data, forecast_data, date_col, value_col)
                            fig = plot_forecast(df, forecast_df, date_col, value_col)
                            st.pyplot(fig)
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat forecasting: {e}")
                st.info("Pastikan nama kolom tanggal dan nilai sudah sesuai dengan file CSV Anda.")
else:
    st.info("Silakan upload file CSV melalui sidebar untuk memulai.")
