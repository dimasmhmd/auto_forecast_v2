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
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data Mentah")
        st.dataframe(df.head())
        try:
            fig_hist, ax_hist = plot_periodic_values_hist(df, value_col)
            st.pyplot(fig_hist)
        except:
            pass

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang melatih model...'):
                    # A. Inisialisasi Class dengan list model
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # B. Jalankan Preprocessing (Nama fungsi yang benar: preprocess_data)
                    # Fungsi ini di modeling.py menerima (data, date_col, value_col)
                    train_df, test_df = forecaster.preprocess_data(df, date_col, value_col)
                    
                    # C. Jalankan Forecast
                    # Fungsi run_forecast di modeling.py menerima (train, test, date_col, value_col)
                    forecaster.run_forecast(train_df, test_df, date_col, value_col)
                    
                    st.success("✅ Prediksi Selesai!")

                    # D. Menampilkan Hasil dari self.stored_models
                    results = forecaster.stored_models
                    
                    for model_name in selected_models:
                        if model_name in results:
                            st.divider()
                            st.subheader(f"📊 Hasil Model: {model_name}")
                            
                            m = results[model_name]
                            c1, c2, c3 = st.columns(3)
                            c1.metric("RMSE", f"{m['rmse']:.2f}")
                            c2.metric("MAE", f"{m['mae']:.2f}")
                            c3.metric("R2 Score", f"{m['r2']:.2f}")
                            
                            # E. Menampilkan Grafik menggunakan fungsi internal modeling.py
                            st.write("Grafik Prediksi vs Aktual:")
                            fig_res = forecaster.plot_results(
                                train_index=train_df[date_col],
                                test_index=test_df[date_col],
                                model_list=[model_name],
                                date_col=date_col,
                                value_col=value_col
                            )
                            st.pyplot(fig_res)
                            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.info("Tips: Pastikan format tanggal di CSV Anda sudah benar (YYYY-MM-DD).")
else:
    st.info("Silakan upload file CSV untuk memulai.")
