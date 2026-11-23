import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import streamlit as st
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from pathlib import Path

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/fashion_resnet50.keras",
        compile=False
    )

model = load_model()

@st.cache_resource
def load_catalog():
    with np.load("data/all_features.npz") as data:
        return data["features"], data["filenames"]

all_features, all_filenames = load_catalog()

@st.cache_resource
def load_neighbors():
    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm='brute',
        metric='euclidean'
    )
    neighbors.fit(all_features)
    return neighbors

neighbors = load_neighbors()


st.title("🛍️ Fashion Recommendation System")

st.write(
    "Beğendiğiniz bir moda ürününün fotoğrafını yükleyin. "
    "Yapay zekâ destekli sistemimiz size benzer ürünleri önersin."
)

uploaded_file = st.file_uploader(
    "Bir ürün fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Seçilen ürün", width=300)

    img_resized = img.resize((224, 224))

    

    img_array = np.array(img_resized, dtype=np.float32)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)

    

    features = model.predict(img_preprocess, verbose=0)

    features = features.flatten()
    features = features / np.linalg.norm(features)

    distances, indices = neighbors.kneighbors(features.reshape(1, -1))

   
    st.subheader("Benzer Ürün Önerileri")
   

    columns = st.columns(5)

    for col, idx in zip(columns, indices[0][:5]):
        img_path = Path("data/images/images") / str(all_filenames[idx])

        with col:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning("Görsel bulunamadı")