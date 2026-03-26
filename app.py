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
    st.error(f"Gagal memuat modul: {e}. Pastikan struktur folder dan file parameters.py sudah benar.")
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
    df[date_col] = pd.to_datetime(df[date_col])
    
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
                    # A. Inisialisasi Class
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # B. Split Data (Manual karena di modeling.py tidak ada preprocess_data)
                    train_size = int(len(df) * 0.8)
                    train_df = df.iloc[:train_size]
                    test_df = df.iloc[train_size:]
                    
                    # C. Latih Model (Fungsi asli di modeling.py Anda adalah train_models)
                    # Train models biasanya menerima data train
                    forecaster.train_models(train_df, date_col, value_col)
                    
                    # D. Prediksi (Fungsi asli di modeling.py Anda adalah predict_models)
                    # Predict models biasanya menerima data test
                    forecaster.predict_models(test_df, date_col, value_col)
                    
                    st.success("✅ Prediksi Selesai!")

                    # E. Menampilkan Hasil
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
                                
                                # Tampilkan grafik hasil
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
else:
    st.info("Silakan upload file CSV untuk memulai.")
