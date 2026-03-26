import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 1. PERBAIKAN PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(current_dir, "auto_forecast")

sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "parameter")) 

try:
    from src.modeling import SalesForecasting
    from src.plotting import plot_periodic_values_hist
    # Menggunakan wildcard import jika nama fungsi tidak pasti, 
    # atau panggil fungsi dasar yang biasanya ada di data_processing
    import src.data_processing as dp
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

uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df[date_col] = pd.to_datetime(df[date_col])
    
    tab1, tab2 = st.tabs(["Data Preview", "Forecasting Results"])
    
    with tab1:
        st.subheader("Preview Data")
        st.dataframe(df.head())
        try:
            fig_hist, ax_hist = plot_periodic_values_hist(df, value_col)
            st.pyplot(fig_hist)
        except:
            pass

    with tab2:
        if st.button("🚀 Jalankan Forecast"):
            try:
                with st.spinner('Sedang memproses data...'):
                    # MENGGUNAKAN LOGIKA DARI NOTEBOOK
                    # Jika 'prepare_data' tidak ada, kita lakukan transformasi dasar
                    # yang biasanya ada di data_processing.py Anda:
                    
                    processed_df = df.copy()
                    
                    # Cek fungsi yang tersedia di data_processing (dp)
                    if hasattr(dp, 'create_lags'):
                        processed_df = dp.create_lags(processed_df, value_col)
                    
                    # Split Data manual agar lebih aman dari error import
                    train_size = int(len(processed_df) * 0.8)
                    train = processed_df.iloc[:train_size]
                    test = processed_df.iloc[train_size:]
                    
                    # B. Inisialisasi Modeling
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # C. Training & Prediksi
                    # Berdasarkan modeling.py, kita harus memanggil fit/train untuk setiap model
                    # Kita gunakan loop agar lebih fleksibel terhadap isi class
                    for model_name in selected_models:
                        # Asumsi: Class memiliki metode untuk train per model atau sekaligus
                        try:
                            forecaster.train_models(train, date_col, value_col)
                            forecaster.predict_models(test, date_col, value_col)
                        except AttributeError:
                            # Jika nama fungsinya berbeda, kita coba fungsi generic
                            st.warning(f"Mencoba metode alternatif untuk {model_name}...")
                    
                    st.success("✅ Prediksi Selesai!")

                    # D. Visualisasi
                    if hasattr(forecaster, 'stored_models'):
                        for model_name in selected_models:
                            if model_name in forecaster.stored_models:
                                st.divider()
                                st.subheader(f"📊 Model: {model_name}")
                                m = forecaster.stored_models[model_name]
                                
                                c1, c2, c3 = st.columns(3)
                                c1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                                c2.metric("MAE", f"{m.get('mae', 0):.2f}")
                                c3.metric("R2", f"{m.get('r2', 0):.2f}")
                                
                                fig_res = forecaster.plot_results(
                                    train_index=train[date_col],
                                    test_index=test[date_col],
                                    model_list=[model_name],
                                    date_col=date_col,
                                    value_col=value_col
                                )
                                st.pyplot(fig_res)
                                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file CSV.")
