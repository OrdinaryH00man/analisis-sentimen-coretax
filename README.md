# Analisis Sentimen Data Media Sosial X pada Website Coretax Menggunakan K-Means Clustering dan IndoBERT

Repositori ini berisi aplikasi web berbasis **Streamlit** untuk menganalisis sentimen pengguna media sosial X (Twitter) terkait implementasi website Coretax. Project ini memanfaatkan **IndoBERT** untuk ekstraksi fitur (*word embedding*) dan **K-Means Clustering** untuk pengelompokan sentimen menjadi 3 kategori: Positif, Negatif, dan Netral.

## 🚀 Fitur Utama
* **Preprocessing Data Otomatis**: Meliputi data cleaning, normalisasi bahasa gaul (menggunakan kamus kata baku), *stopword removal* (Sastrawi/NLTK), dan *stemming*.
* **IndoBERT Embedding**: Representasi teks berbasis konteks yang akurat menggunakan model `indobenchmark/indobert-base-p1`.
* **Analisis Sentimen Real-time**: Masukkan teks tweet apa saja dan sistem langsung memprediksi sentimen serta memberikan deskripsi insight-nya.
* **Visualisasi Data**: Menampilkan hasil clustering berbasis PCA (2D) dan visualisasi kata dominan (*WordCloud*).

---

## 🛠️ Cara Menjalankan Aplikasi di Lokal

### 1. Kloning Repositori
```bash
git clone [https://github.com/username/analisis-sentimen-coretax.git](https://github.com/username/analisis-sentimen-coretax.git)
cd analisis-sentimen-coretax
```

### 2. Buat Virtual Environment & Aktifkan
* **Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

* **Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

---

## Screenshot Aplikasi dan Visualisasi Data

<details>
  <summary><b>1. Halaman Manajemen Dataset & Preprocessing (Klik untuk melihat)</b></summary>
  <br>
  <p align="center">
    <img src="./assets/dataset.png" alt="Manajemen Dataset" width="80%">
    <br><i>Tampilan dataset aktif dan preview upload data baru</i>
  </p>
  <p align="center">
    <img src="./assets/preprocessing.png" alt="Tahapan Preprocessing" width="80%">
    <br><i>Hasil pengujian tiap tahapan teks dari cleaning hingga stemming</i>
  </p>
</details>

<details>
  <summary><b>2. Halaman Hasil Clustering & Evaluasi (Klik untuk melihat)</b></summary>
  <br>
  <p align="center">
    <img src="./assets/scatterplot.png" alt="Hasil PCA dan WordCloud" width="80%">
    <br><i>Grafik sebaran PCA 2D</i>
  </p>
  <p align="center">
    <img src="./assets/distribusi_sentimen.png" alt="Distribusi Tiap Sentimen" width="80%">
    <br><i>Distribusi Sentimen dan Sentimen yang paling dominan</i>
  </p>
  <p align="center">
    <img src="./assets/wordcloud.png" alt="Wordcloud Sentimen" width="80%">
    <br><i>Kata kata yang paling sering muncul pada setiap sentimen</i>
  </p>
</details>

<details>
  <summary><b>3. Halaman Pengujian Sentimen Teks Baru (Klik untuk melihat)</b></summary>
  <br>
  <p align="center">
    <img src="./assets/analisis.png" alt="Uji Sentimen Realtime" width="80%">
    <br><i>Prediksi sentimen realtime menggunakan model IndoBERT & K-Means</i>
  </p>
</details>

---

## 🌐 English Version

<details>
  <summary><b>Click here to read the English version of this documentation</b></summary>
  <br>

  ### Sentiment Analysis of X Social Media Data Using K-Means Clustering and IndoBERT on Coretax Website Case Study

  This repository contains a **Streamlit**-based web application to analyze public sentiments from social media X (Twitter) regarding the implementation of the Indonesian tax system, Coretax. This project utilizes **IndoBERT** (`indobenchmark/indobert-base-p1`) for contextual word embedding and **K-Means Clustering** to group sentiments into 3 categories: Positive, Negative, and Neutral.

  #### 🚀 Key Features
  * **Automated Text Preprocessing**: Data cleaning, slang word normalization, stopword removal, and stemming (Sastrawi/NLTK).
  * **IndoBERT Embedding**: Accurate context-based feature extraction.
  * **Real-time Sentiment Testing**: Predicts and provides descriptions for custom text input.
  * **Data Visualization**: Interactive 2D PCA scatter plot, sentiment distribution, and cluster-based WordClouds.

  #### 🛠️ Local Installation & Setup
  1. Clone the repository: `git clone https://github.com/username/analisis-sentimen-coretax.git`
  2. Install dependencies: `pip install -r requirements.txt`
  3. Run the application: `streamlit run app.py`

  #### 📸 Screenshots & Visualizations
  *The English screenshots share the same structure as the Indonesian section above using interactive dropdowns.*
</details>