import numpy as np
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans

def apply_thresholds(channel, thresholds):
    segmented_image = np.zeros_like(channel, dtype=np.float32)
    bins = [0] + sorted([int(t) for t in thresholds]) + [256]
    for idx in range(len(bins) - 1):
        mask = (channel >= bins[idx]) & (channel < bins[idx + 1])
        if np.any(mask):
            segmented_image[mask] = channel[mask].mean()
    return segmented_image.astype(np.uint8)
    


def otsu_thresholding(image, k, mode="color"):
    if mode == "gray":
        thresholds = threshold_multiotsu(image, classes=k)
        segmented_image = apply_thresholds(image, thresholds)
        return segmented_image,thresholds
    else:
        thresholds = []
        segmented_image = np.zeros_like(image, dtype=np.float32)
        for channel in range(3):
            channel_thresholds = threshold_multiotsu(image[:, :, channel], classes=k)
            segmented_image[:, :, channel] = apply_thresholds(image[:, :, channel], channel_thresholds)
            thresholds.extend(sorted(channel_thresholds))
        return segmented_image.astype(np.uint8),thresholds


def kmeans_thresholding(image, k, mode="color"):
    if mode == "gray":
        x = image.flatten().reshape(-1, 1).astype(np.float32)
        kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
        y_pred = kmeans.fit_predict(x)
        cluster_centers = kmeans.cluster_centers_.flatten()
        segmented_image = cluster_centers[y_pred].reshape(image.shape).astype(np.uint8)
        return segmented_image, sorted(cluster_centers.tolist())
    else:
        segmented_image = np.zeros_like(image, dtype=np.float32)
        thresholds = []
        for channel in range(3):
            x = image[:, :, channel].flatten().reshape(-1, 1).astype(np.float32)
            kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
            y_pred = kmeans.fit_predict(x)
            cluster_centers = kmeans.cluster_centers_.flatten()
            segmented_image[:, :, channel] = cluster_centers[y_pred].reshape(image[:, :, channel].shape)
            thresholds.extend(sorted(cluster_centers.tolist()))
        return segmented_image.astype(np.uint8), thresholds
