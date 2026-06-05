import numpy as np
SMALL_VALUE = 1e-10

def single_channel_kapur(image_channel, thresholds):
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


def kapur_entropy_fitness(image, thresholds, mode="color"):
    if mode == "gray":
        return -single_channel_kapur(image, thresholds)
    else:
        total_entropy = 0.0
        n = len(thresholds) // 3
        idx = 0
        for channel in range(3):
            total_entropy += single_channel_kapur(image[:, :, channel], thresholds[idx:idx + n])
            idx += n
        return -total_entropy 