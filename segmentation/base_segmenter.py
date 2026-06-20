import numpy as np
import sys
import os
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy.swarm_based.ZOA import OriginalZOA
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.physics_based.SA import OriginalSA
from mealpy.math_based.HC import OriginalHC
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.ES import CMA_ES
from algorithms.woazoa import WOAZOA
from segmentation.classical import kmeans_thresholding
from segmentation.classical import otsu_thresholding
from segmentation.problem import SegmentationProblem
from segmentation.classical import apply_thresholds

NATURE_ALGO  = {
    "ga": BaseGA,
    "cmaes": CMA_ES,
    "pso": OriginalPSO,
    "woa": OriginalWOA,
    "gwo": OriginalGWO,
    "zoa": OriginalZOA,
    "woazoa": WOAZOA,
    "hc": OriginalHC,
    "sa": OriginalSA
}

CLASSICAL_ALGO = {
    "otsu": otsu_thresholding,
    "kmeans": kmeans_thresholding
}

DEFAULT_POP_SIZE = 50
DEFAULT_EPOCHS = 500

def segment_image(image, thresholds, mode="color"):
    """
    Apply thresholds to an image to output a segmented image

    Args:
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        thresholds (list of uint8): k-1 threshold values.
        mode (str, optional): 'gray' or 'color' image. Defaults to "color".

    Returns:
        np.ndarray: Segmented image as uint8 np.ndarray with the same shape as the input.
    """
    if mode == "gray":
        return apply_thresholds(image, thresholds)
    else:
        segmented_image = np.zeros_like(image, dtype=np.uint8)
        n = len(thresholds) // 3
        idx = 0
        for channel in range(3):
            segmented_image[:, :, channel] = apply_thresholds(image[:, :, channel], thresholds[idx:idx + n])
            idx += n
        return segmented_image.astype(np.uint8)

def run_segmentation(algorithm_name, image, k, mode="color", params=None):
    """_summary_

    Args:
        algorithm_name (str): Key from NATURE_ALGO or CLASSICAL_ALGO.
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        k (int): Number of segments.
        mode (str, optional): 'gray' or 'color' image. Defaults to "color".
        params (dict, optional): Hyperparameters for metaheuristic algorithms. Defaults to epoch=500, pop_size=80.

    Raises:
        ValueError: If algorithm_name is not in NATURE_ALGO and CLASSICAL_ALGO.

    Returns:
        dict with keys:
            thresholds (list of uint8): k-1 threshold values.
            segmented_image (np.ndarray): Segmented image as uint8 np.ndarray
            fitness: Best Kapur entropy value. None for classical.
            history: Best fitness per epoch. None for classical.
    """
    if params is None:
        params = {}
        
    if algorithm_name in NATURE_ALGO:
        problem = SegmentationProblem(image, k, mode)
        params.setdefault("epoch", DEFAULT_EPOCHS)
        params.setdefault("pop_size", DEFAULT_POP_SIZE)
        model = NATURE_ALGO[algorithm_name](**params)
        model.solve(problem)
        thresholds = np.sort(model.g_best.solution).clip(1,254).astype(int).tolist()
        segmented_image = segment_image(image, thresholds, mode)
        return {
            "thresholds": thresholds,
            "segmented_image": segmented_image,
            "fitness": model.g_best.target.fitness,
            "history": model.history.list_global_best_fit
        }
    
    elif algorithm_name in CLASSICAL_ALGO:
        segmented_image, thresholds = CLASSICAL_ALGO[algorithm_name](image, k, mode)
        return {
            "thresholds": thresholds,
            "segmented_image": segmented_image,
            "fitness": None,
            "history": None
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
