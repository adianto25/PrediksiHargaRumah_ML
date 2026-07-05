import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd

if "prediksi" not in st.session_state:
    st.session_state.prediksi = None

st.set_page_config(
    page_title="Prediksi Harga Rumah Depok",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = joblib.load("models/random_forest_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    lokasi_list = joblib.load("models/lokasi_list.pkl")
    return model, feature_columns, lokasi_list

model, feature_columns, lokasi_list = load_models()

# Calculate feature importance
feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("_", " ")
)

# Sidebar
st.sidebar.title("House Price Prediction")
st.sidebar.write(
    "Prediksi harga rumah Kota Depok menggunakan Machine Learning Random Forest Regression."
)
st.sidebar.divider()
st.sidebar.metric("Algoritma", "Random Forest")
st.sidebar.metric("Akurasi (R² Score)", "0.797")
st.sidebar.metric("Mean Absolute Error", "Rp266 Juta")
st.sidebar.metric("Total Dataset", "19.619 Data")

# Main Title
st.title("Prediksi Harga Rumah Depok")
st.write("Sistem prediksi harga properti tingkat lanjut menggunakan Machine Learning. Silakan masukkan spesifikasi rumah di bawah ini untuk mendapatkan estimasi harga.")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Spesifikasi Rumah")

    kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=20, value=3)
    kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=20, value=2)
    garasi = st.number_input("Jumlah Garasi", min_value=0, max_value=20, value=1)
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=1, value=120)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=1, value=100)
    lokasi = st.selectbox("Lokasi (Kecamatan/Kelurahan)", lokasi_list)

    prediksi_button = st.button("Jalankan Prediksi Harga", type="primary", use_container_width=True)

with col2:
    st.subheader("Hasil Analisis & Prediksi")

    if prediksi_button:
        with st.spinner("Memproses data..."):
            input_data = pd.DataFrame({
                "Kamar Tidur": [kamar_tidur],
                "Kamar Mandi": [kamar_mandi],
                "Garasi": [garasi],
                "Luas Tanah": [luas_tanah],
                "Luas Bangunan": [luas_bangunan],
                "Lokasi": [lokasi]
            })

            input_encoded = pd.get_dummies(input_data, columns=["Lokasi"], drop_first=True)

            input_encoded = input_encoded.reindex(
                columns=feature_columns,
                fill_value=0
            )

            st.session_state.prediksi = model.predict(input_encoded)[0]
            prediksi = st.session_state.prediksi

            if prediksi < 1_000_000_000:
                kategori = "Rumah Menengah"
            elif prediksi < 3_000_000_000:
                kategori = "Rumah Premium"
            else:
                kategori = "Rumah Mewah"

            st.success("Prediksi berhasil dijalankan!")
            st.metric(
                label="Estimasi Harga Rumah",
                value=f"Rp {prediksi:,.0f}".replace(",", ".")
            )

            harga_per_meter = prediksi / luas_bangunan

            st.metric(
                label="Estimasi Harga per m² (Bangunan)",
                value=f"Rp {harga_per_meter:,.0f}".replace(",", ".")
            )

            st.markdown("#### Indikator Harga Proporsional")
            progress = min(prediksi / 5_000_000_000, 1.0)
            st.progress(progress)
            
            st.info(f"**Kategori Properti**: {kategori}")
            
            st.markdown("#### Ringkasan Karakteristik")
            if prediksi < 1_000_000_000:
                st.write("""
                Properti ini diklasifikasikan sebagai tipe **Menengah**. 
                Valuasi sangat bergantung pada rasio luas bangunan dan kondisi lingkungan sekitar. 
                Sangat optimal untuk segmen keluarga muda.
                """)
            elif prediksi < 3_000_000_000:
                st.write("""
                Properti ini diklasifikasikan sebagai tipe **Premium**. 
                Memiliki spesifikasi dan fasilitas yang berada di atas rata-rata pasar perumahan di Depok.
                """)
            else:
                st.write("""
                Properti ini diklasifikasikan sebagai tipe **Mewah / Eksklusif**. 
                Mencerminkan nilai aset yang tinggi dengan keunggulan luas bangunan serta lokasi yang sangat strategis.
                """)
                
            if st.button("Ulangi Prediksi", use_container_width=True):
                st.session_state.prediksi = None
                st.rerun()
    else:
        st.info("Silakan lengkapi form spesifikasi rumah di sebelah kiri, kemudian klik tombol 'Jalankan Prediksi Harga'.")

st.divider()

st.subheader("Informasi Model & Statistik Dataset")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Utama", "Random Forest Regressor")
m2.metric("R² Score", "0.797")
m3.metric("Total Data Pelatihan", "19.619")
m4.metric("Total Fitur", len(feature_columns))

st.divider()

with st.expander("Analisis Fitur (Feature Importance)"):
    st.write("Grafik di bawah ini menunjukkan metrik kepentingan fitur (Feature Importance) yang digunakan oleh model Random Forest. Fitur dengan nilai tinggi memiliki dampak terbesar dalam memprediksi harga rumah di Depok.")
    
    fig, ax = plt.subplots(figsize=(8,4))
    top10 = feature_importance.head(10)
    
    bars = ax.barh(
        top10["Feature"],
        top10["Importance"],
        color="#3B82F6"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Fitur")
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.003,
            bar.get_y() + bar.get_height()/2,
            f"{width:.3f}",
            va="center",
            fontsize=9
        )
        
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

st.caption(
    "Developed by Adianto | UAS Machine Learning | Teknik Informatika | 2026"
)