import numpy as np
import shutil
from pathlib import Path

with np.load("data/all_features.npz") as data:
    features = data["features"]
    filenames = data["filenames"]

indices = np.sort(np.random.default_rng(42).choice(len(filenames), 1000, replace=False))

demo_dir = Path("demo")
(demo_dir / "images").mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    demo_dir / "all_features.npz",
    features=features[indices],
    filenames=filenames[indices]
)

for filename in filenames[indices]:
    shutil.copy2(Path("data/images/images") / filename, demo_dir / "images" / filename)

print("Demo ürün sayısı:", len(indices))