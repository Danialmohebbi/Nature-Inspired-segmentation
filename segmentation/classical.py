import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

def apply_thresholds(channel, thresholds):
    """Apply intensity thresholds to a single channel image devididing the channel into segments.

    Args:
        channel (np.ndarray): 2D np.ndarray of uint8 pixel intensities.
        thresholds (list of uint8): k-1 threshold values.

    Returns:
        np.ndarray: Segmented channel as uint8 np.ndarray with the shape (H,W).
    """
    segmented_image = np.zeros_like(channel, dtype=np.float32)
    bins = [0] + sorted([int(t) for t in thresholds]) + [256]
    for idx in range(len(bins) - 1):
        mask = (channel >= bins[idx]) & (channel < bins[idx + 1])
        if np.any(mask):
            segmented_image[mask] = channel[mask].mean()
    return segmented_image.astype(np.uint8)
    
def recursive_otsu(channel, k):
    """
    Recursively partition a flattened intensity array into k segments using Otsu's binary thresholding at each step.

    Args:
        channel (np.ndarray): Flattened intensity array.
        k (int): Number of segments.

    Returns:
        list: Sorted List of threshold values.
    """
    regions = [channel.flatten()]
    thresholds = []
    
    while len(regions) < k:
        split_idx = max(range(len(regions)), key=lambda i: len(regions[i]))
        regions_pixels = regions[split_idx]
        
        if len(regions_pixels) < 2 or regions_pixels.min() == regions_pixels.max():
            break
        
        local_region_threshold = threshold_otsu(regions_pixels)
        thresholds.append(int(local_region_threshold))
        
        lower = regions_pixels[regions_pixels <= local_region_threshold]
        upper = regions_pixels[regions_pixels > local_region_threshold]
        
        regions.pop(split_idx)
        regions.append(lower)
        regions.append(upper)
    
    return sorted(thresholds)

    

def otsu_thresholding(image, k, mode="color"):
    """
    Segment an image using recursive otsu thresholding.

    Args:
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        k (int): Number of segments.
        mode (str, optional): 'gray' or 'color' image.. Defaults to "color".

    Returns:
        tuple: 
            segmented_image (np.ndarray): Segmented image as uint8 np.ndarray with the same shape as the input image.
            thresholds (list of uint8): k-1 threshold values.
    """
    if mode == "gray":
        thresholds = recursive_otsu(image, k)
        segmented = apply_thresholds(image, thresholds)
        return segmented,thresholds
    else:
        thresholds = []
        segmented = np.zeros_like(image, dtype=np.uint8)
        for channel in range(3):
            channel_thresholds = recursive_otsu(image[:, :, channel], k)
            segmented[:, :, channel] = apply_thresholds(image[:, :, channel], channel_thresholds)
            thresholds.extend(sorted(channel_thresholds))
        return segmented.astype(np.uint8),thresholds


def kmeans_thresholding(image, k, mode="color"):
    """
    Segment an image using KMEANS algorithim. 

    Args:
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        k (int): Number of segments.
        mode (str, optional): 'gray' or 'color' image.. Defaults to "color".

    Returns:
        tuple: 
            segmented_image (np.ndarray): Segmented image as uint8 np.ndarray with the same shape as the input image.
            thresholds (list of uint8): k-1 threshold values.
    """
    if mode == "gray":
        x = image.flatten().reshape(-1,1).astype(np.float32)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y = kmeans.fit_predict(x)
        centroids = kmeans.cluster_centers_.flatten()
        segmented = centroids[y].reshape(image.shape).astype(np.uint8)
        return segmented, sorted(centroids.tolist())
    else:
        segmented = np.zeros_like(image,dtype=np.uint8)
        thresholds = []
        for ch in range(3):
            x = image[:, :, ch].flatten().reshape(-1,1).astype(np.float32)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            y = kmeans.fit_predict(x)
            centroids = kmeans.cluster_centers_.flatten()
            segmented[:, :, ch] = centroids[y].reshape(image[:, :, ch].shape).astype(np.uint8)
            thresholds.extend(sorted(centroids.tolist()))
        return segmented,thresholds


