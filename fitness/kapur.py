import numpy as np
from segmentation.classical import apply_thresholds
from fitness.reconstruction import compute_ssim

LAMBDA = 0.5
L = 256
SMALL_VALUE = 1e-10

def single_channel_kapur(image_channel, thresholds):
    """
    Compute Kapur entropy for a single channel of an image.
    
    Args:
        image_channel (np.ndarray): 2D np.ndarray of uint8 pixel intensities.
        thresholds (list of int): k-1 threshold values.
    
    Returns:
        float: Total kapur entropy for the given image's channel and thresholds.
    """
    hist, _ = np.histogram(image_channel.flatten(), bins=256, range=(0,256))
    hist = hist / hist.sum() 
    
    bins = [0] + sorted(thresholds) + [256]
    channel_entropy = 0.0
    
    for idx in range(len(bins) - 1):
        region = hist[bins[idx]:bins[idx + 1]]
        
        W = float(region.sum())
        if W <= SMALL_VALUE:
            continue
        
        p = region / W
        channel_entropy += float(-np.sum(p * np.log(p + SMALL_VALUE)))
    
    return channel_entropy


def kapur_entropy_fitness(image, thresholds,k, mode="color"):
    """
    Compute negative kapur entropy to be used as a mealpy fitness function

    Args:
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        thresholds (list of uint8): k-1 threshold values.
        mode (str, optional): 'gray' or 'color' image. Defaults to "color".

    Returns:
        float: Negative total kapur entropy
    """
    if mode == "gray":
        return -single_channel_kapur(image, thresholds)
    else:
        total_entropy = 0.0
        n = k - 1
        idx = 0
        for channel in range(3):
            total_entropy += single_channel_kapur(image[:, :, channel], thresholds[idx:idx + n])
            idx += n
        return -total_entropy 

def reconstruct_for_fitness(image, thresholds, k, mode="color"):
    """
    Reconstruct the segmented image from a threshold vector.
    Args:
        image (np.ndarray): uint8 array.
        thresholds (list):  gray is k-1 values. color is 3*(k-1) values.
        k (int): Number of segments.
        mode (str): 'gray' or 'color'.

    Returns:
        np.ndarray: Reconstructed segmented image, uint8.
    """
    if mode == "gray":
        return apply_thresholds(image, thresholds)

    n = k - 1
    segmented = np.zeros_like(image, dtype=np.uint8)
    for ch in range(3):
        segmented[:, :, ch] = apply_thresholds(image[:, :, ch], thresholds[ch*n:(ch+1)*n])
    return segmented

def kapur_ssim(image, thresholds, k, mode="color"):
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

def compute_penality(image):
    img = image.astype(np.float32)
    horizental = np.abs(img[:, :-1] - img[:, 1:]).mean()
    vertical = np.abs(img[:-1, :] - img[1:, :]).mean()
    return float((horizental + vertical) / 2.0) 

def kapur_spatial(image,thresholds, k, mode="color"):
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
