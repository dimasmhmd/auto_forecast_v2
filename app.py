import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import io

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
                    forecaster = SalesForecasting(model_list=selected_models)
                    
                    # Persiapan Data
                    train_size = int(len(df) * 0.8)
                    train_df = df.iloc[:train_size]
                    test_df = df.iloc[train_size:]

                    X_train = train_df.drop(columns=[date_col, value_col])
                    y_train = train_df[value_col]
                    X_test = test_df.drop(columns=[date_col, value_col])
                    
                    if X_train.empty:
                        X_train['dummy'] = np.arange(len(X_train))
                        X_test['dummy'] = np.arange(len(X_train), len(X_train) + len(X_test))

                    # Eksekusi Fit & Predict
                    try:
                        forecaster.fit(X_train, y_train) 
                    except:
                        forecaster.fit(train_df.drop(columns=[date_col]), value_col)
                    
                    try:
                        forecaster.predict(X_test)
                    except:
                        forecaster.predict(test_df.drop(columns=[date_col]))
                    
                    st.success("✅ Prediksi Selesai!")

                    # --- MENAMPILKAN DAN MENYIAPKAN DATA DOWNLOAD ---
                    if hasattr(forecaster, 'stored_models'):
                        # Dictionary untuk menampung semua hasil prediksi untuk Excel
                        all_results_df = test_df[[date_col, value_col]].copy()
                        all_results_df.columns = ['Tanggal', 'Aktual']

                        for model_name in selected_models:
                            if model_name in forecaster.stored_models:
                                st.divider()
                                st.subheader(f"📊 Model: {model_name}")
                                m = forecaster.stored_models[model_name]
                                
                                # Simpan prediksi ke dataframe gabungan
                                all_results_df[f'Prediksi_{model_name}'] = m.get('predictions')

                                # Metrics
                                c1, c2, c3 = st.columns(3)
                                c1.metric("RMSE", f"{m.get('rmse', 0):.2f}")
                                c2.metric("MAE", f"{m.get('mae', 0):.2f}")
                                c3.metric("R2 Score", f"{m.get('r2', 0):.2f}")
                                
                                try:
                                    fig_res = forecaster.plot_results(model_list=[model_name])
                                    st.pyplot(fig_res)
                                except:
                                    st.line_chart(all_results_df.set_index('Tanggal')[[f'Prediksi_{model_name}', 'Aktual']])

                        # --- TOMBOL DOWNLOAD ---
                        st.divider()
                        st.subheader("💾 Download Hasil Prediksi")
                        
                        # Fungsi untuk konversi ke Excel
                        def to_excel(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False, sheet_name='Forecast_Results')
                            return output.getvalue()

                        excel_data = to_excel(all_results_df)

                        st.download_button(
                            label="📥 Download Hasil Prediksi (Excel)",
                            data=excel_data,
                            file_name='hasil_forecast_sales.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                        
                        st.write("Preview Data yang diunduh:")
                        st.dataframe(all_results_df)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan eksekusi: {e}")
else:
    st.info("Silakan unggah file CSV di sidebar.")
