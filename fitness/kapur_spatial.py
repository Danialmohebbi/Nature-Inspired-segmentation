import numpy as np
from fitness.kapur import kapur_entropy_fitness, reconstruct_for_fitness
def compute_penality(image):

    img = image.astype(np.float32)
    horizental = np.abs(img[:, :-1] - img[:, 1:]).mean()
    vertical = np.abs(img[:-1, :] - img[1:, :]).mean()
    return float((horizental + vertical) / 2.0) 

def kapur_spatial_fitness(image,thresholds, k, mode="color"):
    """
        Implement the Spatial Kapur
    Args:
        image (np.ndarray): uint8 array.
        thresholds (list):  gray is k-1 values. color is 3*(k-1) values.
        k (int): Number of segments.
        mode (str): 'gray' or 'color'.

    Returns:
        float: the blended result
    """
    kapur = -kapur_entropy_fitness(image, thresholds, k, mode)
    segmented_image = reconstruct_for_fitness(image, thresholds, k, mode)
    penalty = 0
    if mode == "gray":
        penalty = compute_penality(segmented_image)
    else:
        penalties = []
        for ch in range(3):
            penalties.append(compute_penality(segmented_image[:, :, ch]))
        penalty = float(np.mean(penalties))
    
    return -(kapur / (1 + penalty))
