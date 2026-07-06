import numpy as np
from fitness.kapur import kapur_entropy_fitness,reconstruct_for_fitness
from fitness.reconstruction import compute_ssim
L = 256

def kapur_ssim_fitness(image, thresholds, k, mode="color"):
    """
        Return mix of kapur and ssim
    Args:
        image (np.ndarray): uint8 array.
        thresholds (list):  gray is k-1 values. color is 3*(k-1) values.
        k (int): Number of segments.
        mode (str): 'gray' or 'color'.

    Returns:
        float: the blended result
    """
    kapur = -kapur_entropy_fitness(image, thresholds, k, mode)
    n_channels = 3 if mode == "color" else 1
    kapur_norm = kapur / (n_channels * k * np.log(L)) 
    segmented_image = reconstruct_for_fitness(image, thresholds, k, mode)
    ssim_value = compute_ssim(image, segmented_image, mode)
    return -((1 - LAMBDA) * kapur_norm + LAMBDA * float(ssim_value))  # type: ignore[reportArgumentType]