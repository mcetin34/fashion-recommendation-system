# Fashion Recommendation System

A visual fashion recommendation application built with **TensorFlow, ResNet50, scikit-learn, and Streamlit**. Upload a product photo to discover visually similar items from a catalog of **44,441 fashion images**.

## Features

- Uses an ImageNet-pretrained **ResNet50** backbone and `GlobalMaxPool2D` for feature extraction.
- Represents every catalog image as an **L2-normalized, 2,048-dimensional vector**.
- Uses **NearestNeighbors** with Euclidean distance to retrieve visually similar products.
- Displays **five product recommendations** in a Streamlit interface.
- Supports local Python execution and a Docker-based deployment.

## How it works

1. A user uploads a JPG, JPEG, or PNG image.
2. The image is resized to **224 × 224** and preprocessed for ResNet50.
3. The saved model extracts and normalizes its 2,048 visual features.
4. `NearestNeighbors` searches the precomputed feature catalog.
5. Streamlit displays the five closest matches from the image directory.

**Model note:** The ResNet50 backbone uses pretrained ImageNet weights; it was **not trained from scratch or fine-tuned** on this dataset. The catalog features were precomputed in Google Colab in batches of 32 and saved in 4,000-image chunks before being merged.

## Project structure

```text
.
├── app.py
├── Fashion_Batch_Processing.ipynb
├── kontrol.py
├── requirements.txt
├── dockerfile
├── .gitignore
├── models/                        # Download separately; not committed
│   └── fashion_resnet50.keras
└── data/                          # Download separately; not committed
    ├── all_features.npz
    └── images/
        └── images/
            ├── 10000.jpg
            └── ...
```

## Required data and model files

Large assets are **not included in the Git repository**. Before starting the application, prepare these three assets:

| Asset | Expected location | Source |
|---|---|---|
| Saved ResNet50 feature extractor (~93 MB) | `models/fashion_resnet50.keras` | Project GitHub Release asset — **not yet published** |
| Precomputed feature catalog (~281 MB) | `data/all_features.npz` | Project GitHub Release asset — **not yet published** |
| Fashion product images | `data/images/images/*.jpg` | [Kaggle — Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) |

**Image extraction:** Download the **Small** dataset (not the large full-resolution dataset). Extract its `images` folder so that a file such as `data/images/images/10000.jpg` exists. Do not rename individual image files: filenames must correspond to the precomputed feature catalog.

**Current availability:** Until the project Release assets are published, `fashion_resnet50.keras` and `all_features.npz` must be provided from the original project files. Downloading the repository alone is therefore **not sufficient** to build or run the full application.

## Run locally

Requires Python and the three assets above.

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Open <http://localhost:8502>.

## Run with Docker (full catalog)

Requires Docker Desktop / Docker Engine and the same assets in the paths shown above **before building**. The current `dockerfile` copies the model, catalog, and images into the Docker image.

```bash
docker build -t fashion-recommender .
docker run --rm -p 8504:8501 --name fashion-recommender-app fashion-recommender
```

Open <http://localhost:8504>.

> The full Docker image includes the product images and may be large. Model/catalog downloads are **not automated yet**; a future setup script can download the Release assets after their public URLs exist.

## Notebook and reproducibility

`Fashion_Batch_Processing.ipynb` documents the Kaggle data loading, pretrained ResNet50 feature extraction, batch processing, normalization, NearestNeighbors retrieval, sample recommendations, and model export. `kontrol.py` was used to verify and merge the `.npz` feature chunks.

The saved `all_features.npz` contains both the feature matrix (`44,441 × 2,048`) and the ordered image filenames. The filenames must remain aligned with feature rows.

## Limitations

- The Small dataset contains low-resolution product photos (typically **60 × 80 pixels**), so displayed recommendations may appear blurry when enlarged.
- Similarity is based on visual features, not guaranteed agreement on brand, category, or product identity.
- If an uploaded image already exists in the catalog, that image may appear among the recommendations.
- Quantitative recommendation-quality metrics have not yet been evaluated.

## Dataset acknowledgement

Dataset: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), published on Kaggle by Param Aggarwal. Review the dataset terms and underlying image rights before redistributing product photographs. This repository is intended to share project code; the images are obtained separately from the dataset source.

## Publication status

- [x] Google Colab feature extraction and notebook
- [x] Streamlit recommendations using the full local catalog
- [x] Full local Docker build and runtime test
- [ ] Publish model and catalog files as GitHub Release assets
- [ ] Add automated asset setup and a lightweight online demo
- [ ] Add a public demo video / screenshots
