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
    image_directory = ROOT / "data" / "images" / "images"
    if image_directory.is_dir() and any(image_directory.glob("*.jpg")):
        print("[SKIP] Product images are already present")
        return

    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit("Install Kaggle downloader first: pip install kagglehub") from error

    print("[DOWNLOAD] Kaggle: fashion-product-images-small")
    kagglehub.dataset_download(
        "paramaggarwal/fashion-product-images-small",
        output_dir=str(ROOT / "data" / "images"),
    )
    if not image_directory.is_dir() or not any(image_directory.glob("*.jpg")):
        raise RuntimeError(
            "Kaggle download completed, but images were not found in data/images/images. "
            "Check the downloaded dataset folder structure before running the app."
        )
    print("[OK] Product images are ready")


def main() -> None:
    for name, destination in ASSETS.items():
        download_asset(name, destination)
    download_images()
    print("\nAll assets are ready. You can build Docker or run Streamlit.")


if __name__ == "__main__":
    main()
