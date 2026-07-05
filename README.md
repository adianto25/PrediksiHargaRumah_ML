# 🏠 Prediksi Harga Rumah di Depok Menggunakan Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-Educational-green)

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi Machine Learning untuk memprediksi harga rumah di Kota Depok berdasarkan karakteristik properti menggunakan algoritma **Random Forest Regression**.

Aplikasi dibangun menggunakan **Python**, **Scikit-Learn**, dan **Streamlit**, kemudian dideploy sehingga dapat digunakan secara online.

---

## 🌐 Demo Aplikasi

**Streamlit**

https://uas-machine-learning-prediksi-harga-rumah-depok-akyvtuz6m8nj5n.streamlit.app/

---

## 🎯 Tujuan

- Memprediksi harga rumah berdasarkan spesifikasi rumah.
- Membandingkan performa model Machine Learning.
- Mengembangkan aplikasi prediksi berbasis web menggunakan Streamlit.
- Menerapkan proses Machine Learning mulai dari preprocessing hingga deployment.

---

## 📊 Dataset

Dataset yang digunakan merupakan data penjualan rumah di Kota Depok.

Jumlah data awal

- 24.561 data

Jumlah data setelah cleaning

- 19.619 data

Target Prediksi

- Harga Rumah

Fitur yang digunakan

- Jumlah Kamar Tidur
- Jumlah Kamar Mandi
- Jumlah Garasi
- Luas Tanah
- Luas Bangunan
- Lokasi

---

## ⚙️ Tahapan Machine Learning

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Menghapus Missing Value
4. Menghapus Data Duplikat
5. Menghapus Outlier
6. Encoding Variabel Lokasi
7. Train Test Split
8. Training Model
9. Evaluasi Model
10. Deployment

---

## 🤖 Algoritma

Model utama yang digunakan adalah

**Random Forest Regression**

Model pembanding

- Linear Regression

---

## 📈 Hasil Evaluasi

| Model | MAE | R² Score |
|--------|---------:|---------:|
| Linear Regression | Rp326.133.837 | 0.725 |
| Random Forest Regression | Rp266.100.206 | **0.797** |

Model terbaik yang dipilih adalah **Random Forest Regression**.

---

## 📌 Feature Importance

Berdasarkan hasil pelatihan model, fitur yang paling berpengaruh terhadap harga rumah adalah

1. Luas Bangunan
2. Luas Tanah
3. Kamar Mandi
4. Kamar Tidur
5. Lokasi
6. Garasi

---

## 🖥️ Tampilan Aplikasi

Fitur aplikasi meliputi

- Input spesifikasi rumah
- Prediksi harga rumah
- Informasi model
- Statistik dataset
- Visualisasi Feature Importance
- Ringkasan hasil prediksi

---

## 📁 Struktur Project

```
UAS_MachineLearning
│
├── dataset
├── images
├── models
├── notebook
├── presentation
├── report
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ Menjalankan Project

Clone repository

```bash
git clone https://github.com/zakyilhamf647586-spec/UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok.git
```

Masuk folder project

```bash
cd UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok
```

Install library

```bash
pip install -r requirements.txt
```

Jalankan aplikasi

```bash
streamlit run app.py
```

---

## 🛠️ Tools

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- VS Code
- Git
- GitHub

---

## 👨‍💻 Author

**Zaky Ilham Ferdiansyah**

Program Studi Teknik Informatika

Universitas Dian Nuswantoro

2026
