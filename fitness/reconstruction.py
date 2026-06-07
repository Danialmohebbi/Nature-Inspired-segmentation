import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import piq
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

DEVICE = torch.device("cpu")
PATCH_SIZE = 8
SMALL_NUMBER = 1e-10

def convert_to_tensor(image, mode):
    """
    Convert an uint8 np.ndarray to a float tensor.

    Args:
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        mode (str): 'gray' or 'color' image.

    Raises:
        ValueError: if mode is not 'gray' or 'color'

    Returns:
        torch.tensor: Float tensor on DEVICE in [0,1]
    """
    image = torch.tensor(image, dtype=torch.float32) / 255.0
    if mode == "color":
        image = image.permute(2, 0, 1).unsqueeze(0)
    elif mode == "gray":
        image = image.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid mode for tensor conversion. Use 'color' or 'gray'.")
    return image
    
def convert_to_luminance(tensor):
    """
    Convert a RGB tensor to grayscale luminance.

    Args:
        tensor (torch.tensor): Float tensor of shape (1,3,H,W).

    Returns:
        torch.tensor: Luminance tensor of shape (1,1,H,W).
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)
    return torch.sum(tensor * weights, dim=1, keepdim=True)

def compute_mse(original, reconstructed, mode=None):
    """
    Compute Mean Squared Error between original and segmented image.

    Args:
        original (np.ndarray): uint8 np.ndarray representing the original image.
        reconstructed (np.ndarray): uint8 np.ndarray representing the segmented image.
        mode (str, optional): 'gray' or 'color' image.. Defaults to None.

    Returns:
        float: Mean Squared Error accross all pixels.
    """
    return mean_squared_error(original.flatten().astype(np.float32), reconstructed.flatten().astype(np.float32))

def compute_psnr(original, reconstructed, mode=None):
    """
    Compute Peak Signal to Noise Ratio between original and segmented image.
    Args:
        original (np.ndarray): uint8 np.ndarray representing the original image.
        reconstructed (np.ndarray): uint8 np.ndarray representing the segmented image.
        mode (str, optional): 'gray' or 'color' image.. Defaults to None.

    Returns:
        Float: PSNR value.
    """
    return psnr(original, reconstructed, data_range=255)

def compute_ssim(original, reconstructed, mode="color"):
    """
    Compute Structual Similarity Index between original and segmented image.

    Args:
        original (np.ndarray): uint8 np.ndarray representing the original image.
        reconstructed (np.ndarray): uint8 np.ndarray representing the segmented image.
        mode (str, optional): 'gray' or 'color' image.. Defaults to 'color'.

    Raises:
        ValueError: If mode is not 'gray' or 'color'.

    Returns:
        float: SSIM value.
    """
    if mode == "color":
        return ssim(original, reconstructed, data_range=255, channel_axis=2)
    elif mode == "gray":
        return ssim(original, reconstructed, data_range=255)
    else:
        raise ValueError("Invalid mode for computing SSIM. Use 'color' or 'gray'.")
    
def compute_fsim(original, reconstructed, mode="color"):
    """
    Compute Feature Similarity Index between original and segmented image.

    Args:
        original (np.ndarray): uint8 np.ndarray representing the original image.
        reconstructed (np.ndarray): uint8 np.ndarray representing the segmented image.
        mode (str, optional): 'gray' or 'color' image.. Defaults to 'color'.

    Raises:
        ValueError: If mode is not 'gray' or 'color'.

    Returns:
        float: FSIM value.
    """
    if mode == "color":
        original_tensor = convert_to_tensor(original, mode="color")
        reconstructed_tensor = convert_to_tensor(reconstructed, mode="color")
                
    elif mode == "gray":
        original_tensor = convert_to_tensor(original, mode="gray")
        reconstructed_tensor = convert_to_tensor(reconstructed, mode="gray")
        
        original_tensor = original_tensor.repeat(1, 3, 1, 1)
        reconstructed_tensor = reconstructed_tensor.repeat(1, 3, 1, 1)

    else:
        raise ValueError("Invalid mode for computing FSIM. Use 'color' or 'gray'.")
    
    return piq.fsim(original_tensor, reconstructed_tensor, data_range=1.0).item()
    

def compute_qilv(original, reconstructed, mode = "color"):
    """
    Compute Quality Index based on Local Variance between original and segmented image.
    
    Args:
        original (np.ndarray): uint8 np.ndarray representing the original image.
        reconstructed (np.ndarray): uint8 np.ndarray representing the segmented image.
        mode (str, optional): 'gray' or 'color' image.. Defaults to 'color'.

    Raises:
        ValueError: If mode is not 'gray' or 'color'.

    Returns:
        float: QILV value.

    """
    if mode == "gray":
        original_tensor = convert_to_tensor(original, mode="gray")
        reconstructed_tensor = convert_to_tensor(reconstructed, mode="gray")
    elif mode == "color":
        original_tensor = convert_to_tensor(original, mode="color")
        reconstructed_tensor = convert_to_tensor(reconstructed, mode="color")
        
        original_tensor = convert_to_luminance(original_tensor)
        reconstructed_tensor = convert_to_luminance(reconstructed_tensor)
    else:
        raise ValueError("Invalid mode for computing QILV. Use 'color' or 'gray'.")
    
    
    unfold = torch.nn.Unfold(kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    
    varience_original = unfold(original_tensor).var(dim=1)
    varience_reconstructed = unfold(reconstructed_tensor).var(dim=1)
    mean_original = varience_original.mean()
    mean_reconstructed = varience_reconstructed.mean()
    standard_deviation_original = varience_original.std()
    standard_deviation_reconstructed = varience_reconstructed.std()
    std_dev__both = ((varience_original - mean_original) * (varience_reconstructed - mean_reconstructed)).mean()
    
    result_1 = (2 * mean_original * mean_reconstructed + SMALL_NUMBER) / (mean_original ** 2 + mean_reconstructed ** 2 + SMALL_NUMBER)
    result_2 = (2 * standard_deviation_original * standard_deviation_reconstructed + SMALL_NUMBER) / (standard_deviation_original ** 2 + standard_deviation_reconstructed ** 2 + SMALL_NUMBER)
    result_3 = (std_dev__both + SMALL_NUMBER) / (standard_deviation_original * standard_deviation_reconstructed + SMALL_NUMBER)
    
    return (result_1 * result_2 * result_3).item()


def compute_all_metrics(original, reconstructed, mode="color"):
    """
    Compute all image quality metrics between original and segmented image.
    
    Args:
        original (np.ndarray): uint8 np.ndarray representing the original image.
        reconstructed (np.ndarray): uint8 np.ndarray representing the segmented image.
        mode (str, optional): 'gray' or 'color' image.. Defaults to 'color'.


    Returns:
        dict with keys:
            MSE (float): Mean Squared Error. Lower is better.
            PSNR (float): Peak Signal to Ratio. Higher is better.
            SSIM (float): Structual Similarity Index. Higher is better.
            FSIM (float): Feature Similarity Index. Higher is better.
            QILV (float): Quality Index based on Local Varience. Higher is better.
    """
    mse = compute_mse(original, reconstructed, mode)
    psnr_value = compute_psnr(original, reconstructed, mode)
    ssim_value = compute_ssim(original, reconstructed, mode)
    fsim_value = compute_fsim(original, reconstructed, mode)
    qilv_value = compute_qilv(original, reconstructed, mode)
    
    return {
        "MSE": mse,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "FSIM": fsim_value,
        "QILV": qilv_value
    }