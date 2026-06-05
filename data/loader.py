import os
import cv2
import functools 

TRAIN_DATA = "train"
TEST_DATA = "test"
BASE_DIR = "data\\splits\\"

def load_image(path, mode="color"):
    image = cv2.imread(path)
    if mode == "grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif mode == "color":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Invalid mode. Use 'color' or 'grayscale'.")
    return image

def get_image_paths(split = "train"):
    if split == "train":
        data_dir = BASE_DIR + TRAIN_DATA
    elif split == "test":
        data_dir = BASE_DIR + TEST_DATA
    else:
        raise ValueError("Invalid split. Use 'train' or 'test'.")
    
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    return image_paths


