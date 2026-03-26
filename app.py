import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 1. PERBAIKAN PATH (Paling Penting)
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(current_dir, "auto_forecast")

# Tambahkan folder root, src, dan parameter ke sys.path
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "parameter")) # Agar 'import parameters' berhasil

try:
    from src.modeling import SalesForecasting
    from src.plotting import plot_periodic_values_hist
    # Import data processing sesuai notebook
    from src.data_processing import prepare_data, split_data 
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.info("Pastikan folder 'src' dan 'parameter' ada di dalam folder 'auto_forecast'.")
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
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df[date_col] = pd.to_datetime(df[date_col])
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data")
        st.dataframe(df.head())
        fig_hist, ax_hist = plot_periodic_values_hist(df, value_col)
        st.pyplot(fig_hist)

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Memproses data & Training...'):
                    # A. Preprocessing menggunakan src.data_processing
                    # Mengikuti alur di example_notebook.ipynb
                    processed_data = prepare_data(df, date_col, value_col)
                    train, test = split_data(processed_data)
                    
                    # B. Inisialisasi Modeling
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # C. Training & Prediksi
                    # Sesuai isi modeling.py, kita panggil fit dan predict (atau fungsi utamanya)
                    forecaster.train(train, date_col, value_col)
                    forecaster.predict(test, date_col, value_col)
                    
                    st.success("✅ Prediksi Selesai!")

                    # D. Tampilkan Visualisasi
                    results = forecaster.stored_models
                    for model_name in selected_models:
                        if model_name in results:
                            st.divider()
                            st.subheader(f"📊 Model: {model_name}")
                            
                            # Metrics
                            m = results[model_name]
                            c1, c2, c3 = st.columns(3)
                            c1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                            c2.metric("MAE", f"{m.get('mae', 0):.2f}")
                            c3.metric("R2 Score", f"{m.get('r2', 0):.2f}")
                            
                            # Plotting menggunakan fungsi internal modeling.py
                            fig_res = forecaster.plot_results(
                                train_index=train[date_col],
                                test_index=test[date_col],
                                model_list=[model_name],
                                date_col=date_col,
                                value_col=value_col
                            )
                            st.pyplot(fig_res)
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan eksekusi: {e}")
else:
    st.info("Silakan unggah file CSV di sidebar.")
