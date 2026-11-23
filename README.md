# Fashion Recommendation System

A visual fashion recommendation application built with **TensorFlow, ResNet50, scikit-learn, and Streamlit**. Upload a product photo to discover visually similar items from a catalog of **44,441 fashion images**.

## Live Demo

🚀 [Try the Fashion Recommendation System](https://fashion-recommendation-system-2.streamlit.app/)

Upload a fashion product image and discover five visually similar products using Deep Learning.

**Online Demo:** 1,000 products  
**Full Docker Version:** 44,441 products


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
├── setup_assets.py
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

## Required data and model files

Large assets are **not included in the Git repository**. The application requires the following three assets:

| Asset | Expected location | Source |
|---|---|---|
| Saved ResNet50 feature extractor (~93 MB) | `models/fashion_resnet50.keras` | [GitHub Release v1.0.0](https://github.com/mcetin34/fashion-recommendation-system/releases/tag/v1.0.0) |
| Precomputed feature catalog (~281 MB) | `data/all_features.npz` | [GitHub Release v1.0.0](https://github.com/mcetin34/fashion-recommendation-system/releases/tag/v1.0.0) |
| Fashion product images | `data/images/images/*.jpg` | [Kaggle — Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) |

### Automatic asset setup

The project includes `setup_assets.py` to simplify installation.

Run:

```bash
pip install kagglehub
python setup_assets.py
```

The setup script checks whether the required assets already exist. If they are missing, it attempts to download:

- The saved ResNet50 model from GitHub Releases.
- The precomputed feature catalog from GitHub Releases.
- The product images from Kaggle.

Existing assets are skipped to avoid unnecessary downloads.

### Manual setup

Alternatively, download the model and feature catalog from [GitHub Releases](https://github.com/mcetin34/fashion-recommendation-system/releases/tag/v1.0.0) and place them in their expected directories.

Download the **Small** version of the Kaggle dataset, not the full-resolution version.

Extract the images so that a file such as `data/images/images/10000.jpg` exists.

**Important:** Do not rename the images. Their filenames must match the precomputed feature catalog.

**Note:** The automatic setup has been tested with existing local assets. A complete download from a clean environment has not yet been verified.

## Run locally

Requires Python and the three assets above.

```bash
pip install -r requirements.txt
pip install kagglehub
python setup_assets.py
streamlit run app.py --server.port 8502
```

Open <http://localhost:8502>.

## Run with Docker (full catalog)

Requires Docker Desktop / Docker Engine and the same assets in the paths shown above **before building**. The current `dockerfile` copies the model, catalog, and images into the Docker image.

```bash
pip install kagglehub
python setup_assets.py

docker build -t fashion-recommender .
docker run --rm -p 8504:8501 --name fashion-recommender-app fashion-recommender
```

Open <http://localhost:8504>.



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
- [x] Publish model and catalog files as GitHub Release assets
- [x] Add automatic asset setup script
- [ ] Verify asset downloads in a clean environment
- [ ] Deploy a lightweight online demo
- [ ] Add a public demo video / screenshots
