import os
import cv2

BASE_DIR   = "C:\\Users\\daniy\\OneDrive\\Desktop\\Thesis\\data\\images"
SPLITS_DIR = "C:\\Users\\daniy\\OneDrive\\Desktop\\Thesis\\data\\splits"

SPLIT_FOLDERS = {
    "train": ["train", "val"],
    "test":  ["test"],
}


def load_image(path, mode="color"):
    """
    Load an image from disk and convert to the specified color mode.

    Args:
        path (str):  Absolute path to image file.
        mode (str):  'color' → RGB uint8 array (H, W, 3).
                     'gray'  → grayscale uint8 array (H, W).

    Returns:
        np.ndarray: uint8 image array.
    """
    image = cv2.imread(path)
    if mode == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif mode == "color":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("mode must be 'color' or 'gray'")


def _find_image(filename):
    """
    Search for an image filename across all BSD500 subfolders.

    Args:
        filename (str): Image filename e.g. '54005.jpg'

    Returns:
        str: Full absolute path to the image.

    Raises:
        FileNotFoundError: If image not found in any subfolder.
    """
    for subfolder in ["train", "val", "test"]:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image {filename} not found in BSD500 directories")


def get_image_paths(split="train"):
    """
    Get image paths for a given split using the txt file.

    Reads filenames from data/splits/train.txt or test.txt,
    then resolves each filename to its full path in BSD500.

    Args:
        split (str): 'train' or 'test'.

    Returns:
        list of str: Absolute paths to all images in the split.
    """
    txt_path = os.path.join(SPLITS_DIR, f"{split}.txt")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Split file not found: {txt_path}")

    with open(txt_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    image_paths = []
    for filename in filenames:
        try:
            path = _find_image(filename)
            image_paths.append(path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    return image_paths


if __name__ == "__main__":
    train = get_image_paths("train")
    test  = get_image_paths("test")
    print(f"Train: {len(train)}")
    print(f"Test:  {len(test)}")
    overlap = set(train) & set(test)
    print(f"Overlap: {len(overlap)}")  # must be 0