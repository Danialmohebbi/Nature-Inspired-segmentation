import numpy as np
from skimage.filters import threshold_multiotsu
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
    


def otsu_thresholding(image, k, mode="color"):
    """
    Segment an image using multi-level otsu thresholding.

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
        thresholds = threshold_multiotsu(image, classes=k)
        segmented_image = apply_thresholds(image, thresholds)
        return segmented_image,thresholds
    else:
        thresholds = []
        segmented_image = np.zeros_like(image, dtype=np.uint8)
        for channel in range(3):
            channel_thresholds = threshold_multiotsu(image[:, :, channel], classes=k)
            segmented_image[:, :, channel] = apply_thresholds(image[:, :, channel], channel_thresholds)
            thresholds.extend(sorted(channel_thresholds))
        return segmented_image.astype(np.uint8),thresholds


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
        x = image.flatten().reshape(-1, 1).astype(np.float32)
        kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
        y_pred = kmeans.fit_predict(x)
        cluster_centers = kmeans.cluster_centers_.flatten()
        segmented_image = cluster_centers[y_pred].reshape(image.shape).astype(np.uint8)
        return segmented_image, sorted(cluster_centers.tolist())
    else:
        segmented_image = np.zeros_like(image, dtype=np.uint8)
        thresholds = []
        for channel in range(3):
            x = image[:, :, channel].flatten().reshape(-1, 1).astype(np.uint8)
            kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
            y_pred = kmeans.fit_predict(x)
            cluster_centers = kmeans.cluster_centers_.flatten()
            segmented_image[:, :, channel] = cluster_centers[y_pred].reshape(image[:, :, channel].shape)
            thresholds.extend(sorted(cluster_centers.tolist()))
        return segmented_image.astype(np.uint8), thresholds



