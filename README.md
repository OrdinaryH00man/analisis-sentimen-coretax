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