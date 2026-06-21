import os
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"

SPLIT_FOLDERS = {
    "train": ["train", "val"],
    "test": ["test"],
}


def load_image(path, mode="color"):
    """
    Load an image from disk and convert to the specified color mode.
    Args:
        path (str(: Path to image file.
        mode (str): 'gray' or 'color' image type.
    Returns:
        np.ndarray: uint8 image array.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cv2 could not read image: {path}")
    if mode == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif mode == "color":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("mode must be 'color' or 'gray'")


def _find_image(file):
    """
    Search for an image filename across all BSD500 subfolders.
    Args:
        filename (str): Image filename.
    Returns:
        Path (str): Full path to the image.
    Raises:
        FileNotFoundError: If image not found in any subfolder.
    """
    filename, mode = file.split(',')
    path = BASE_DIR / filename
    if path.exists():
        return path,mode
    raise FileNotFoundError(f"Image {filename} not found in Dataset")


def get_image_paths(split="train"):
    """
    Get image paths for a given split using the txt file.
    Args:
        split (str): 'train' or 'test'.
    Returns:
        list of Path: Paths to all images in the split.
    """
    txt_path = SPLITS_DIR / f"{split}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {txt_path}")

    with open(txt_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    image_paths = []
    for filename in filenames:
        try:
            path,mode = _find_image(filename)
            image_paths.append((path,mode))
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return image_paths