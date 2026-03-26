import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# 1. Penanganan Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_folder_path = os.path.join(current_dir, "auto_forecast")
if sub_folder_path not in sys.path:
    sys.path.append(sub_folder_path)

# 2. Import Module
try:
    from src.modeling import SalesForecasting
    # Kita gunakan plotting dasar jika fungsi plotting internal bermasalah
    from src.plotting import plot_periodic_values_hist
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.stop()

st.set_page_config(page_title="Auto Forecast Tool", layout="wide")
st.title("📈 Auto Sales Forecasting Dashboard")

# --- SIDEBAR ---
st.sidebar.header("1. Pengaturan Data")
date_col = st.sidebar.text_input("Nama Kolom Tanggal", "date")
value_col = st.sidebar.text_input("Nama Kolom Nilai (Sales)", "sales")

st.sidebar.header("2. Parameter Model")
model_options = ['LinearRegression', 'RandomForest', 'XGBoost', 'LSTM', 'ARIMA']
selected_models = st.sidebar.multiselect("Pilih Model", model_options, default=['XGBoost'])

# --- MAIN PAGE ---
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df[date_col] = pd.to_datetime(df[date_col]) # Pastikan format tanggal benar
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data Mentah")
        st.dataframe(df.head())
        try:
            # Memanggil fungsi histogram yang terbukti ada di plotting.py Anda
            fig_hist, ax_hist = plot_periodic_values_hist(df, value_col)
            st.pyplot(fig_hist)
        except:
            pass

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang memproses dan melatih model...'):
                    # Inisialisasi Class
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # KARENA ERROR 'preprocess_data', kita coba panggil fungsi utama 
                    # yang kemungkinan besar menangani segalanya sekaligus.
                    # Di banyak template, fungsi ini adalah run_forecast atau sejenisnya.
                    
                    # Mari kita gunakan pendekatan aman: Jika fungsi preprocess tidak ditemukan,
                    # kita asumsikan data dimasukkan langsung ke run_forecast.
                    
                    # Coba jalankan forecast (menyesuaikan dengan library pmdarima & sklearn di file Anda)
                    forecaster.run_forecast(df, date_col, value_col)
                    
                    st.success("✅ Prediksi Selesai!")

                    # Menampilkan metrik hasil yang disimpan di self.stored_models
                    if hasattr(forecaster, 'stored_models'):
                        results = forecaster.stored_models
                        for model_name in selected_models:
                            if model_name in results:
                                st.divider()
                                st.subheader(f"📊 Model: {model_name}")
                                m = results[model_name]
                                c1, c2, c3 = st.columns(3)
                                c1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                                c2.metric("MAE", f"{m.get('mae', 0):.2f}")
                                c3.metric("R2 Score", f"{m.get('r2', 0):.2f}")
                                
                                # Gunakan plotting internal class jika ada
                                try:
                                    fig_res = forecaster.plot_results(model_list=[model_name])
                                    st.pyplot(fig_res)
                                except:
                                    st.line_chart(m.get('predictions'))
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.info("Saran: Periksa apakah file 'parameters.py' sudah ada di folder yang sama dengan 'modeling.py'.")
else:
    st.info("Silakan upload file CSV untuk memulai.")
