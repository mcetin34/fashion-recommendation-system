"""Download the external assets needed by Fashion Recommendation System.

Usage:
    pip install kagglehub
    python setup_assets.py

Files already present are preserved; the script never overwrites them.
"""

from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import zipfile


ROOT = Path(__file__).resolve().parent
RELEASE_BASE = (
    "https://github.com/mcetin34/"
    "fashion-recommendation-system/releases/download/v1.0.0"
)
ASSETS = {
    "fashion_resnet50.keras": ROOT / "models" / "fashion_resnet50.keras",
    "all_features.npz": ROOT / "data" / "all_features.npz",
}


def download_asset(name: str, destination: Path) -> None:
    if destination.is_file() and zipfile.is_zipfile(destination):
        print(f"[SKIP] Already exists: {destination.relative_to(ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    print(f"[DOWNLOAD] GitHub Release: {name}")
    try:
        req = Request(f"{RELEASE_BASE}/{name}", headers={"User-Agent": "fashion-recommender-setup"})
        with urlopen(req, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not zipfile.is_zipfile(temporary):
            raise ValueError(f"Downloaded file is not a valid archive: {name}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[OK] {destination.relative_to(ROOT)}")


def download_images() -> None:
    root = ROOT / "data" / "images"
    image_directory = root / "images"

    def check_images():
        # Fotoğraflar bir üst klasördeyse doğru yere taşı.
        if root.exists() and any(root.glob("*.jpg")):
            image_directory.mkdir(parents=True, exist_ok=True)
            for photo in root.glob("*.jpg"):
                photo.rename(image_directory / photo.name)

        return sum(1 for _ in image_directory.glob("*.jpg"))

    count = check_images()

    if count == 44441:
        print(f"[SKIP] {count} product images are already present")
        return

    if count > 0:
        raise SystemExit(f"[ERROR] Incomplete image catalog: {count}/44441")

    try:
        import kagglehub

        print("[DOWNLOAD] Kaggle: fashion-product-images-small")
        kagglehub.dataset_download(
            "paramaggarwal/fashion-product-images-small",
            output_dir=str(root),
        )

    except Exception as error:
        raise SystemExit(
            f"[WARNING] Kaggle download failed: {type(error).__name__}\n"
            "Download images manually from:\n"
            "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small\n"
            "Extract the images into data/images/images/ and rerun setup_assets.py"
        ) from None

    count = check_images()

    if count != 44441:
        raise SystemExit(f"[ERROR] Expected 44441 images, found {count}")

    print("[OK] Product images are ready")


def main() -> None:
    for name, destination in ASSETS.items():
        download_asset(name, destination)
    download_images()
    print("\nAll assets are ready. You can build Docker or run Streamlit.")


if __name__ == "__main__":
    main()
