from pathlib import Path

import numpy as np

data = np.load("data/fashion_0_3999.npz")

print(data["features"].shape)
print(len(data["filenames"]))

files = list(Path("data").glob("fashion_*.npz"))

print("Dosya sayısı:", len(files))
print("Toplam ürün:", sum(np.load(f)["features"].shape[0] for f in files))

files = sorted(files, key=lambda f: int(f.stem.split("_")[1]))

print([f.name for f in files])

all_features = np.concatenate([np.load(f)["features"] for f in files])

all_filenames = np.concatenate([np.load(f)["filenames"] for f in files])

print(all_features.shape)
print(all_filenames.shape)

np.savez_compressed(
    "data/all_features.npz",
    features=all_features,
    filenames=all_filenames
)

from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(
    n_neighbors=6,
    algorithm='brute',
    metric='euclidean'
)

neighbors.fit(all_features)

distances, indices = neighbors.kneighbors([all_features[0]])

print("Benzer ürün indeksleri:", indices[0])
print("Ürün dosyaları:", all_filenames[indices[0]])

from PIL import Image

img = Image.open("data/images/images/34754.jpg")
print(img.size)