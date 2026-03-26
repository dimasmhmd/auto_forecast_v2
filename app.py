import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 1. PENANGANAN PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(current_dir, "auto_forecast")

sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "parameter")) 

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
                with st.spinner('Sedang melatih model...'):
                    # 1. Inisialisasi Class
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # 2. Persiapan Data untuk Model (MENGHINDARI ERROR DATETIME)
                    # Kita buat copy agar data asli tetap utuh untuk plotting
                    df_model = df.copy()
                    
                    # Simpan index tanggal untuk keperluan plotting nanti
                    train_size = int(len(df_model) * 0.8)
                    
                    # Pisahkan fitur (X) dan target (y) atau hapus kolom non-numerik
                    # Model XGBoost/RandomForest tidak bisa menerima kolom datetime
                    train_data = df_model.iloc[:train_size].drop(columns=[date_col])
                    test_data = df_model.iloc[train_size:].drop(columns=[date_col])
                    
                    # 3. EKSEKUSI FIT
                    # Mengirim data yang SUDAH BERSIH dari kolom tanggal
                    try:
                        forecaster.fit(train_data, value_col) 
                    except Exception as fit_err:
                        # Jika fit gagal, mungkin ia butuh index sebagai tanggal
                        # Kita coba kirim data dengan tanggal sebagai index
                        train_idx = df_model.iloc[:train_size].set_index(date_col)
                        forecaster.fit(train_idx, value_col)
                    
                    # 4. PREDICT
                    try:
                        forecaster.predict(test_data)
                    except:
                        test_idx = df_model.iloc[train_size:].set_index(date_col)
                        forecaster.predict(test_idx)
                    
                    st.success("✅ Prediksi Selesai!")

                    # 5. Menampilkan Hasil
                    if hasattr(forecaster, 'stored_models'):
                        for model_name in selected_models:
                            if model_name in forecaster.stored_models:
                                st.divider()
                                st.subheader(f"📊 Model: {model_name}")
                                m = forecaster.stored_models[model_name]
                                
                                c1, c2, c3 = st.columns(3)
                                c1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                                c2.metric("MAE", f"{m.get('mae', 0):.2f}")
                                c3.metric("R2 Score", f"{m.get('r2', 0):.2f}")
                                
                                # Visualisasi
                                try:
                                    fig_res = forecaster.plot_results(model_list=[model_name])
                                    st.pyplot(fig_res)
                                except:
                                    # Fallback jika plot internal gagal
                                    st.line_chart(m.get('predictions'))
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan eksekusi: {e}")
                st.info("Saran: Kolom tanggal otomatis dihapus sebelum masuk ke model XGBoost.")
else:
    st.info("Silakan unggah file CSV di sidebar.")
