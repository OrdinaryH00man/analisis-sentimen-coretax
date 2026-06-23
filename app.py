import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import torch
import matplotlib.pyplot as plt
import pickle
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

# ================= KONFIGURASI GRAFIK UNTUK PRINT =================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'font.size': 14,          
    'axes.labelsize': 16,     
    'axes.titlesize': 18,     
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

#================= Dataset =================

@st.cache_data
def load_initial_dataset():
    df = pd.read_csv("Sentimen_Coretax.csv")
    df = df.dropna(subset=["full_text"])
    return df

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Analisis Sentimen IndoBERT & K-Means",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "login" not in st.session_state:
    st.session_state["login"] = False

if "base_df" not in st.session_state:
    st.session_state["base_df"] = load_initial_dataset()

# ================= Global CSS (MODERN ACADEMIC THEME) =================
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 20px !important;
        color: #212529 !important;
        background-color: #FFFFFF !important;
    }

    .stApp {
        background-color: #F8F9FA !important; 
    }

    section[data-testid="stSidebar"] {
        width: 360px !important;
        background-color: #FFFFFF !important;
        border-right: 1px solid #dee2e6;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }

    .page-title {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #004aad !important;
        text-align: center;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .page-subtitle {
        font-size: 22px !important;
        font-weight: 500;
        color: #6c757d !important;
        text-align: center;
        margin-bottom: 40px;
    }

    h1, h2, h3, h4 {
        color: #004aad !important;
        font-weight: 700 !important;
    }

    .card {
        padding: 25px !important;
        border-radius: 8px;
        background-color: #FFFFFF; 
        border-left: 6px solid #004aad !important; 
        border-top: 1px solid #dee2e6;
        border-right: 1px solid #dee2e6;
        border-bottom: 1px solid #dee2e6;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; 
    }

    [data-testid="stMetricValue"] {
        font-size: 60px !important;
        font-weight: 900 !important;
        color: #004aad !important; 
    }

    [data-testid="stMetricLabel"] {
        font-size: 22px !important;
        font-weight: 600;
        color: #495057 !important;
    }

    [data-testid="stDataFrame"] th {
        font-size: 24px !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        background-color: #004aad !important; 
        border-bottom: 3px solid #003380 !important;
        vertical-align: middle !important;
        text-align: center !important;
    }

    [data-testid="stDataFrame"] td {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #212529 !important;
        background-color: #ffffff !important;
        border-bottom: 2px solid #dee2e6 !important;
        line-height: 1.6 !important;
    }
    
    [data-testid="stDataFrame"] [data-testid="stTable"] th[role="rowheader"] {
        font-size: 22px !important;
        font-weight: 900 !important;
        color: #004aad !important;
        background-color: #f8f9fa !important;
    }

    button {
        background-color: #004aad !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; 
    }
    button:hover {
        background-color: #003380 !important; 
        transform: translateY(-2px); 
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #212529 !important;
        border: 2px solid #ced4da !important;
        border-radius: 6px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #004aad !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 74, 173, 0.25) !important;
    }

    .vega-embed text {
        fill: #212529 !important;
        font-size: 16px !important;
    }

    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    header[data-testid="stHeader"] button {
        background-color: transparent !important; 
        color: #333333 !important; 
        border: none !important;
        box-shadow: none !important;
    }

    header[data-testid="stHeader"] button:hover {
        background-color: #f0f0f0 !important; 
        color: #000000 !important;
        transform: none !important; 
        box-shadow: none !important;
    }

    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ================= NLTK =================
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# ================= STOPWORD KHUSUS WORDCLOUD (DOMAIN CORETAX) =================
custom_domain_stopwords = {
    "pajak", "coretax", "djp", "akun", "website", "login",
    "lapor", "bayar", "wp", "kpp", "pph", "ppn",
    "sistem", "aplikasi", "data", "online",
    "tahun", "hari", "orang", "buat", "bikin",
    "isi", "masuk", "daftar", "pakai", "guna", "min", "aju",
    "gue", "kak", "hai", "allah", "npwp", "wajib",
    "pajakku", "dong", "nih", "deh", "loh", "yaudah", "yuk",
    "dong", "kalo", "kalo", "gak", "ga", "nggak", "nya", "si", "ya", "sila",
    "aktivasi", "kakak", "mohon", "bantu", "bantuan", "paja", "kode", "giat"
}

# ================= STEMMER =================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ================= NORMALIZATION =================
@st.cache_data
def load_normalization_dict():
    kamus = pd.read_excel("kamuskatabaku.xlsx")
    return dict(zip(kamus["tidak_baku"], kamus["kata_baku"]))

normalized_words = load_normalization_dict()

def normalize_text(text):
    return " ".join([normalized_words.get(word, word) for word in text.split()])

# ================= PREPROCESSING FUNCTIONS =================
def datacleaning(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', str(text))
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r'[?|$|.|@#%^/&*=!_:")(-+,]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.replace('\n', ' ')
    return text.strip()

def case_folding(text):
    return text.lower()

def tokenizing(text):
    return word_tokenize(text)

def filtering(tokens):
    return " ".join([w for w in tokens if w not in stop_words])

def stemming(text):
    return stemmer.stem(text)

def full_preprocess_df(df):
    df = df.copy()

    df["cleaning"] = df["full_text"].apply(datacleaning)
    df = df.drop_duplicates(subset=["cleaning"])

    df["case_folding"] = df["cleaning"].apply(case_folding)
    df["normalisasi"] = df["case_folding"].apply(normalize_text)
    df["tokenizing"] = df["normalisasi"].apply(tokenizing)
    df["stopword"] = df["tokenizing"].apply(filtering)
    df["stemming"] = df["stopword"].apply(stemming)

    return df

def full_preprocess_text(text):
    text = datacleaning(text)
    text = case_folding(text)
    text = normalize_text(text)
    tokens = tokenizing(text)
    text = filtering(tokens)
    text = stemming(text)
    return text

# ================= INDOBERT =================
@st.cache_resource
def load_indobert():
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_indobert()

def get_embedding(text):
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

@st.cache_resource
def build_base_model(df):
    st.info("Membangun model awal (hanya dilakukan sekali)...")

    df = df.copy()
    df["cleaning"] = df["full_text"].apply(datacleaning)
    df.drop_duplicates(subset=["cleaning"], inplace=True)
    df["case_folding"] = df["cleaning"].apply(case_folding)
    df["normalisasi"] = df["case_folding"].apply(normalize_text)
    df["tokenizing"] = df["normalisasi"].apply(tokenizing)
    df["stopword"] = df["tokenizing"].apply(filtering)
    df["stemming"] = df["stopword"].apply(stemming)

    embeddings = []
    for text in df["stemming"]:
        embeddings.append(get_embedding(text))
    X = np.vstack(embeddings)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init="auto"
    )
    df["cluster_label"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df["component_1"] = components[:, 0]
    df["component_2"] = components[:, 1]

    return df, kmeans, pca

# ================= WORDCLOUD FUNCTION (High Contrast) =================
def generate_wordcloud_small(text_series):
    text = " ".join(text_series)

    wc = WordCloud(
        width=400,
        height=400,
        background_color="white", 
        colormap="magma",         
        stopwords=stop_words.union(custom_domain_stopwords),
        collocations=False,
        max_words=150,
        contour_width=1,         
        contour_color='black'     
    ).generate(text)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

def preprocess_single_text(text):
    text = datacleaning(text)
    text = case_folding(text)
    text = normalize_text(text)
    tokens = tokenizing(text)
    text = filtering(tokens)
    text = stemming(text)
    return text

@st.cache_resource
def load_base_model():
    with open("base_model.pkl", "rb") as f:
        return pickle.load(f)

# ================= LOAD DATASET AWAL =================
if "base_df" not in st.session_state:
    st.session_state["base_df"] = load_initial_dataset()

BASE_MODEL_PATH = "base_model.pkl"

if not os.path.exists(BASE_MODEL_PATH):
    base_df, base_kmeans, base_pca = build_base_model(st.session_state["base_df"])
    with open(BASE_MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "data": base_df,
                "kmeans": base_kmeans,
                "pca": base_pca
            },
            f
        )

base_model = load_base_model()
st.session_state["base_df"] = base_model["data"]

base_df = base_model["data"]
base_kmeans = base_model["kmeans"]
base_pca = base_model["pca"]

# ================= SIDEBAR =================
st.sidebar.title("Analisis Sentimen IndoBERT & K-Means Clustering pada Website Coretax")

if "menu" not in st.session_state:
    st.session_state["menu"] = "Home"

if "login" not in st.session_state:
    st.session_state["login"] = False

with st.sidebar.container(border=True):

    # ===== HOME (SELALU ADA) =====
    if st.button("🏠 Home", use_container_width=True):
        st.session_state["menu"] = "Home"

    # ===== BELUM LOGIN =====
    if not st.session_state["login"]:
        if st.button("🔐 Login", use_container_width=True):
            st.session_state["menu"] = "Login"

    # ===== SUDAH LOGIN =====
    if st.session_state["login"]:
        if st.button("📂 Dataset", use_container_width=True):
            st.session_state["menu"] = "Dataset"

        if st.button("⚙️ Preprocessing", use_container_width=True):
            st.session_state["menu"] = "Preprocessing"

        if st.button("📊 Result", use_container_width=True):
            st.session_state["menu"] = "Result"

        if st.button("📈 Analisis Sentimen", use_container_width=True):
            st.session_state["menu"] = "Analisis Sentimen"

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["login"] = False
            st.session_state["menu"] = "Home"
            st.rerun()

menu = st.session_state["menu"]

# ================= HOME =================
if menu == "Home":
    st.markdown("""
    <div class="fade-in">
        <div class="page-title">
            📊 Sistem Analisis Sentimen Coretax
        </div>
        <div class="page-subtitle">
            Menggunakan IndoBERT dan K-Means Clustering
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ===== DEMO BOX (LIGHT MODE) =====
    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: center;
        ">
            <div style="
                width: 70%;
                background-color: #ffffff;
                padding: 30px;
                border: 2px solid #000000; /* Border hitam tebal */
                border-radius: 12px;
            ">
                <h3 style="text-align:center; color: #000000; border-bottom: 2px solid #000; padding-bottom: 10px;">
                    Panduan Penggunaan
                </h3>
                <ol style="font-size:20px; line-height:2.0; color: #000000; font-weight: 500;">
                    <li><b>Login</b> ke dalam sistem menggunakan akun admin.</li>
                    <li>Unggah dataset baru (CSV) pada halaman <b>Dataset</b> jika diperlukan.</li>
                    <li>Cek tahapan pembersihan data di halaman <b>Preprocessing</b>.</li>
                    <li>Lihat visualisasi clustering dan evaluasi model di halaman <b>Result</b>.</li>
                    <li>Gunakan halaman <b>Analisis Sentimen</b> untuk menguji teks ulasan baru.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================= LOGIN =================
elif menu == "Login":
    st.markdown("""
    <div class="page-title">🔐 Login Admin</div>
    <div class="page-subtitle">Akses pengelolaan sistem</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:

        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Login", use_container_width=True):
            if username == "farid" and password == "admin":
                st.session_state["login"] = True
                st.session_state["menu"] = "Dataset"
                st.success("Login berhasil")
                st.rerun()
            else:
                st.error("Username atau password salah")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= DATASET =================
elif menu == "Dataset":
    st.markdown("""
    <div class="page-title">📂 Manajemen Dataset</div>
    <div class="page-subtitle">
    Dataset aktif dan penggantian dataset untuk model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card fade-in">
    <h4>📄 Dataset Aktif</h4>
    <p>Dataset ini digunakan sebagai dasar pembentukan model IndoBERT dan K-Means.</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== DATASET AKTIF =====
    st.subheader("Dataset Aktif")

    if "base_df" in st.session_state:
        active_df = st.session_state["base_df"]
        st.info(f"Jumlah data aktif: {len(active_df)}")
        st.dataframe(active_df.head(20))
    else:
        st.warning("Dataset belum tersedia")

    st.divider()

    # ===== GANTI DATASET =====
    st.subheader("Upload Dataset Baru")

    uploaded_file = st.file_uploader(
        "Upload dataset CSV baru",
        type=["csv"],
        help="Dataset ini akan menggantikan dataset lama"
    )

    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        st.write("Preview dataset baru:")
        st.dataframe(new_df.head(10))

        st.warning(
            "Dataset ini akan menggantikan dataset sebelumnya dan model akan dibangun ulang."
        )

        if st.button("🔄 Bangun Model dari Dataset Baru"):
            with st.spinner("Membangun model baru..."):
                new_df, new_kmeans, new_pca = build_base_model(new_df)

                # SIMPAN KE SESSION
                st.session_state["base_df"] = new_df
                st.session_state["base_kmeans"] = new_kmeans
                st.session_state["base_pca"] = new_pca

            st.success("Dataset & model berhasil diganti")

# ================= PREPROCESSING =================
elif menu == "Preprocessing":
    st.markdown("""
    <div class="page-title">⚙️ Hasil Preprocessing</div>
    <div class="page-subtitle">
    Tahapan preprocessing yang digunakan dalam penelitian
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card fade-in">
    <p>
    Preprocessing dilakukan satu kali pada dataset awal dan digunakan
    untuk membangun model analisis sentimen.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ===== FLAG VISIBILITAS =====
    if "show_preprocessing" not in st.session_state:
        st.session_state["show_preprocessing"] = False

    # ===== TOMBOL =====
    if not st.session_state["show_preprocessing"]:
        if st.button("⚙️ Tampilkan Hasil Preprocessing"):
            st.session_state["show_preprocessing"] = True


    # ===== JIKA BELUM DITEKAN =====
    if not st.session_state["show_preprocessing"]:
        st.warning("Klik tombol di atas untuk menampilkan hasil preprocessing.")
        st.stop()

    # ===== DATA (SUDAH DIPROSES) =====
    df = st.session_state["base_df"]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📝 Teks Asli",
        "🧹 Cleaning",
        "🔡 Case Folding",
        "🔁 Normalisasi",
        "✂️ Tokenizing",
        "🚫 Stopword Removal",
        "🌱 Stemming"
    ])

    with tab1:
        st.subheader("Teks Asli")
        st.dataframe(df[["full_text"]].head(50))

    with tab2:
        st.subheader("Hasil Data Cleaning")
        st.dataframe(df[["full_text", "cleaning"]].head(50))

    with tab3:
        st.subheader("Hasil Case Folding")
        st.dataframe(df[["cleaning", "case_folding"]].head(50))
    with tab4:
        st.subheader("Hasil Normalisasi")
        st.dataframe(df[["case_folding", "normalisasi"]].head(50))

    with tab5:
        st.subheader("Hasil Tokenizing")
        st.dataframe(df[["normalisasi", "tokenizing"]].head(50))
    with tab6:
        st.subheader("Hasil Stopword Removal")
        st.dataframe(df[["tokenizing", "stopword"]].head(50))

    with tab7:
        st.subheader("Hasil Stemming (Teks Akhir)")
        st.dataframe(df[["stopword", "stemming"]].head(50))
# ================= RESULT =================
elif menu == "Result":
    st.markdown("""
    <div class="page-title">📊 Hasil Analisis Sentimen</div>
    <div class="page-subtitle">
    Visualisasi hasil IndoBERT dan K-Means Clustering
    </div>
    """, unsafe_allow_html=True)

    data = base_model["data"]

    # ================= RINGKASAN =================
    st.markdown(
        """
        Halaman ini menampilkan hasil analisis sentimen menggunakan
        **IndoBERT sebagai representasi teks** dan **K-Means Clustering**
        sebagai metode pengelompokan sentimen.
        """
    )

    st.divider()

    # ================= SILHOUETTE SCORE =================
    st.subheader("Evaluasi Kualitas Clustering")

    X = np.vstack(data["embeddings"].values)
    silhouette_avg = silhouette_score(X, data["cluster_label"])

    st.markdown(
        f"""
        <div style="text-align:center;">
            <h3>Silhouette Score</h3>
            <h1 style="color:#2E86C1;">{silhouette_avg:.4f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card fade-in">
    <p style="text-align: justify;">
    Silhouette Score digunakan untuk mengevaluasi kualitas hasil clustering dengan
    mengukur kedekatan data dalam satu cluster dan perbedaannya dengan cluster lain.
    Nilai berada pada rentang <b>-1 hingga 1</b>, di mana nilai yang lebih tinggi
    menunjukkan pemisahan cluster yang semakin baik. Pada aplikasi ini, nilai
    Silhouette Score memberikan gambaran umum kualitas pengelompokan sentimen
    berdasarkan representasi teks menggunakan IndoBERT.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ================= DISTRIBUSI CLUSTER =================
    st.subheader("Distribusi Sentimen")

    cluster_count = data["sentiment"].value_counts()
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
    cluster_count.plot(kind="bar", ax=ax_bar)

    ax_bar.set_title("Distribusi Sentimen")
    ax_bar.set_xlabel("Sentimen")
    ax_bar.set_ylabel("Jumlah Data")
    ax_bar.set_xticklabels(cluster_count.index, rotation=0)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig_bar)
    
    st.divider()

    # ================= PCA SCATTER =================
    st.subheader("Visualisasi Clustering (PCA 2 Dimensi)")

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        data["component_1"],
        data["component_2"],
        c=data["cluster_label"],
        alpha=0.7,
        s=40
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Sebaran Data Hasil K-Means Clustering")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.grid(True)
    st.pyplot(fig)

    st.divider()

    # ================= WORDCLOUD =================
    st.subheader("WordCloud Tiap Cluster")

    sentiment_mapping = {
        0: "Positif",
        1: "Negatif",
        2: "Netral"
    }

    cols = st.columns(3)

    for idx, cluster_id in enumerate(sorted(data["cluster_label"].unique())):
        with cols[idx]:
            st.markdown(
                f"<div style='text-align:center; font-weight:600;'>"
                f"Cluster {cluster_id} ({sentiment_mapping[cluster_id]})"
                f"</div>",
                unsafe_allow_html=True
            )

            cluster_data = data[data["cluster_label"] == cluster_id]

            generate_wordcloud_small(cluster_data["stemming"])

    st.divider()

    # ================= CONTOH DATA =================
    st.subheader("Hasil Analisis Sentimen")

    st.dataframe(
        data[
            [
                "full_text",
                "sentiment",
                "cluster_label",
                "component_1",
                "component_2"
            ]
        ].head(50)
    )

        # ================= ANALISIS SENTIMEN =================
elif menu == "Analisis Sentimen":
    st.markdown("""
    <div class="page-title">📈 Analisis Sentimen Teks Baru</div>
    <div class="page-subtitle">
    Analisis sentimen menggunakan model yang telah dibangun
    </div>
    """, unsafe_allow_html=True)

    input_text = st.text_area("Masukkan teks sentimen:")

    if st.button("Analisis Sentimen"):
        if input_text.strip() == "":
            st.warning("Teks tidak boleh kosong")
        else:
            # preprocessing
            processed_text = full_preprocess_text(input_text)

            # embedding indobertnya
            embedding_raw = get_embedding(processed_text)

            # prediksi cluster 
            embedding = np.array([embedding_raw]).astype(base_kmeans.cluster_centers_.dtype)
            cluster = base_kmeans.predict(embedding)[0]

            # PCA 2-dimensinya
            pca_point = base_pca.transform(embedding)[0]

            # ===== TAMPILKAN HASIL PREPROCESSING =====
            st.text_area(
                "Teks Setelah Preprocessing",
                processed_text,
                height=100,
                disabled=False
            )

            # 5️⃣ MAPPING SENTIMEN (SAMA DENGAN RESULT)
            sentiment_mapping = {
                0: "Positif",
                1: "Negatif",
                2: "Netral"
            }

            sentiment = sentiment_mapping[cluster]

            # ===== OUTPUT =====
            st.success(f"Hasil Sentimen: **{sentiment}**")
            st.write("Cluster:", cluster)
            if sentiment == "Positif":
                st.write("Pengguna ini memiliki opini positif terhadap layanan Coretax. " \
                "Pengguna ini merasa puas dengan fitur, kemudahan penggunaan, atau layanan pelanggan yang diberikan oleh Coretax.")
            elif sentiment == "Negatif":
                st.write("Pengguna ini memiliki opini negatif terhadap layanan Coretax. " \
                "Hal ini bisa disebabkan oleh berbagai faktor seperti masalah teknis, pengalaman pengguna yang buruk, atau ketidakpuasan terhadap layanan yang diberikan oleh Coretax.")
            else:
                st.write("Pengguna ini memiliki opini netral terhadap layanan Coretax. " \
                "Pengguna tidak memiliki perasaan yang kuat baik positif maupun negatif terhadap layanan Coretax, atau memiliki pengalaman yang campur aduk dengan layanan tersebut.")