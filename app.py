import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd

if "prediksi" not in st.session_state:
    st.session_state.prediksi = None

st.set_page_config(
    page_title="Prediksi Harga Rumah Depok",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load("models/random_forest_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
lokasi_list = joblib.load("models/lokasi_list.pkl")

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

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
}
.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #F3FFF1;
    border: 1px solid #B8F26B;
}
.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #F8F8F8;
    border: 1px solid #E5E5E5;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏠 House Price Prediction")
st.sidebar.write(
    "Prediksi harga rumah Kota Depok menggunakan Machine Learning Random Forest Regression."
)
st.sidebar.markdown("---")
st.sidebar.metric("Model", "Random Forest")
st.sidebar.metric("R² Score", "0.797")
st.sidebar.metric("MAE", "Rp266 Juta")
st.sidebar.metric("Dataset", "19.619")

st.markdown('<div class="main-title">🏠 Prediksi Harga Rumah Depok</div>', unsafe_allow_html=True)
st.write("Masukkan spesifikasi rumah untuk mendapatkan estimasi harga berdasarkan model Machine Learning.")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Spesifikasi Rumah")

    kamar_tidur = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=20, value=3)
    kamar_mandi = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=20, value=2)
    garasi = st.number_input("Jumlah Garasi", min_value=0, max_value=20, value=1)
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=1, value=120)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=1, value=100)
    lokasi = st.selectbox("Lokasi", lokasi_list)

    prediksi_button = st.button("🔍 Prediksi Harga", use_container_width=True)

with col2:
    st.subheader("💰 Hasil Prediksi")

    if prediksi_button:
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
            kategori = "🟢 Rumah Menengah"

        elif prediksi < 3_000_000_000:
            kategori = "🟡 Rumah Premium"

        else:
            kategori = "🔴 Rumah Mewah"

        st.success("✅ Prediksi berhasil dilakukan!")
        st.metric(
            label="💰 Estimasi Harga Rumah",
            value=f"Rp {prediksi:,.0f}".replace(",", "."),
            delta="Hasil Prediksi Model"
        )

        harga_per_meter = prediksi / luas_bangunan

        st.metric(
            "📐 Harga per m²",
            f"Rp {harga_per_meter:,.0f}".replace(",", ".")
        )

        st.markdown("### 📈 Indikator Harga")
        progress = min(prediksi / 5_000_000_000, 1.0)
        st.progress(progress)
        st.info(f"Kategori : {kategori}")
        st.markdown("### 📋 Ringkasan Analisis")

        if st.button("🔄 Prediksi Lagi", use_container_width=True):
            st.session_state.prediksi = None
            st.rerun()

        if prediksi < 1_000_000_000:
            st.write("""
            Rumah ini termasuk kategori **menengah**.
            Harga terutama dipengaruhi oleh luas bangunan, luas tanah, dan lokasi.
            Cocok untuk kebutuhan hunian keluarga.
            """)

        elif prediksi < 3_000_000_000:
            st.write("""
            Rumah ini termasuk kategori **premium**.
            Memiliki spesifikasi yang cukup tinggi sehingga bernilai lebih dibanding rata-rata rumah di Depok.
            """)

        else:
            st.write("""
            Rumah ini termasuk kategori **mewah**.
            Nilai properti sangat tinggi dengan karakteristik luas bangunan dan lokasi yang unggul.
            """)

    else:
        st.markdown(
            """
            <div class="info-box">
            Masukkan data rumah di sebelah kiri, lalu klik tombol prediksi.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

st.subheader("📊 Informasi Model")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Model", "Random Forest")
m2.metric("R² Score", "0.797")
m3.metric("MAE", "Rp266 Juta")
m4.metric("Data Training", "19.619")

st.markdown("---")

st.markdown("---")

st.subheader("📊 Statistik Dataset")

d1, d2, d3 = st.columns(3)

d1.metric("Jumlah Data", "19.619")
d2.metric("Jumlah Lokasi", len(lokasi_list))
d3.metric("Jumlah Fitur", len(feature_columns))

st.markdown("---")

with st.expander("📈 Lihat 10 Feature Paling Berpengaruh"):

    fig, ax = plt.subplots(figsize=(7,3.8))

    top10 = feature_importance.head(10)

    bars = ax.barh(
        top10["Feature"],
        top10["Importance"]
    )

    ax.invert_yaxis()

    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")

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

st.markdown("---")

st.caption(
    "Developed by Zaky Ilham Ferdiansyah | UAS Machine Learning | Teknik Informatika | Universitas Dian Nuswantoro | 2026"
)