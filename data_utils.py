"""Dataset discovery and deterministic radiograph preprocessing."""

from pathlib import Path

from torchvision import datasets, transforms


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def find_dataset_root(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    candidates = [p for p in [path, *path.rglob("chest_xray")] if (p / "train" / "NORMAL").is_dir()
                  and (p / "train" / "PNEUMONIA").is_dir() and (p / "test").is_dir()]
    if not candidates:
        raise FileNotFoundError("Expected chest_xray/{train,test}/{NORMAL,PNEUMONIA} under " + str(path))
    return min(candidates, key=lambda p: (len(p.parts), str(p)))


def image_folder(root: Path, split: str):
    return datasets.ImageFolder(root / split, transform=IMAGE_TRANSFORM)
