import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_mse(original, reconstructed):
    return mean_squared_error(original.flatten().astype(np.float32), reconstructed.flatten().astype(np.float32))

def compute_psnr(original, reconstructed):
    return psnr(original, reconstructed, data_range=255)

def compute_ssim(original, reconstructed, mode="color"):
    if mode == "color":
        return ssim(original, reconstructed, data_range=255, channel_axis=2)
    
    return ssim(original, reconstructed, data_range=255)

