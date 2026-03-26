import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# 1. Penanganan Path (Sesuaikan dengan struktur folder Anda)
current_dir = os.path.dirname(os.path.abspath(__file__))
sub_folder_path = os.path.join(current_dir, "auto_forecast")
if sub_folder_path not in sys.path:
    sys.path.append(sub_folder_path)

# 2. Import Module
try:
    from src.modeling import SalesForecasting
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

# --- MAIN PAGE: UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data Mentah")
        st.dataframe(df.head())
        # Menampilkan histogram dari plotting.py
        fig_hist, ax_hist = plot_periodic_values_hist(df, value_col)
        st.pyplot(fig_hist)

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang melatih model dan menghitung prediksi...'):
                    # A. Inisialisasi Class
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # B. Jalankan Proses (Sesuaikan dengan alur modeling.py)
                    # Kita asumsikan data dimasukkan melalui fungsi yang ada di class tersebut
                    # Catatan: Pastikan method di bawah ini sesuai dengan isi modeling.py Anda
                    forecaster.data = df
                    forecaster.date_col = date_col
                    forecaster.value_col = value_col
                    
                    forecaster.pre_process()
                    forecaster.run_forecast() # Jalankan training & prediksi
                    
                    st.success("✅ Prediksi Selesai!")

                    # C. MENAMPILKAN HASIL (PENTING)
                    # modeling.py menyimpan hasil di self.stored_models
                    results = forecaster.stored_models
                    
                    for model_name in selected_models:
                        if model_name in results:
                            st.subheader(f"Hasil Model: {model_name}")
                            
                            # Menampilkan Metrik (RMSE, MAE, R2)
                            m = results[model_name]
                            col1, col2, col3 = st.columns(3)
                            col1.metric("RMSE", f"{m['rmse']:.2f}")
                            col2.metric("MAE", f"{m['mae']:.2f}")
                            col3.metric("R2 Score", f"{m['r2']:.2f}")
                            
                            # Menampilkan Grafik (Menggunakan fungsi plot internal modeling.py)
                            st.write("Grafik Prediksi vs Aktual:")
                            fig_res = forecaster.plot_results(model_list=[model_name])
                            st.pyplot(fig_res)
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.info("Pastikan nama kolom di sidebar sama persis dengan yang ada di file CSV.")
else:
    st.info("Silakan upload file CSV untuk memulai.")
