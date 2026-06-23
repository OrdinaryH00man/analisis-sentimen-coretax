import pandas as pd
import numpy as np
import re
import nltk
import torch
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ================= NLTK =================
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# ================= STEMMER =================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ================= NORMALIZATION =================
kamus = pd.read_excel("kamuskatabaku.xlsx")
normalized_words = dict(zip(kamus["tidak_baku"], kamus["kata_baku"]))

def normalize_text(text):
    return " ".join([normalized_words.get(word, word) for word in text.split()])

# ================= PREPROCESSING =================
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

# ================= LOAD DATASET =================
print("Loading dataset...")
df = pd.read_csv("Sentimen_Coretax.csv")

# ================= PREPROCESSING PIPELINE =================
print("Running preprocessing...")
df["cleaning"] = df["full_text"].apply(datacleaning)
df.drop_duplicates(subset=["cleaning"], inplace=True)

df["case_folding"] = df["cleaning"].apply(case_folding)
df["normalisasi"] = df["case_folding"].apply(normalize_text)
df["tokenizing"] = df["normalisasi"].apply(tokenizing)
df["stopword"] = df["tokenizing"].apply(filtering)
df["stemming"] = df["stopword"].apply(stemming)

# ================= INDOBERT =================
print("Loading IndoBERT...")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
model.eval()

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generating embeddings...")
df["embeddings"] = df["stemming"].apply(get_embedding)

X = np.vstack(df["embeddings"].values)

# ================= K-MEANS =================
print("Running K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["cluster_label"] = kmeans.fit_predict(X)

# ================= PCA =================
print("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df["component_1"] = X_pca[:, 0]
df["component_2"] = X_pca[:, 1]

# ================= SENTIMENT LABEL =================
sentiment_mapping = {
    0: "Positive",
    1: "Negative",
    2: "Neutral"
}
df["sentiment"] = df["cluster_label"].map(sentiment_mapping)

# ================= SAVE BASE MODEL =================
print("Saving base model...")
with open("base_model.pkl", "wb") as f:
    pickle.dump(
        {
            "data": df,
            "kmeans": kmeans,
            "pca": pca
        },
        f
    )

print("✅ Base model rebuilt successfully!")