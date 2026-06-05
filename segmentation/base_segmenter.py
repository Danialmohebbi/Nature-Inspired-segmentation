import numpy as np
import sys
import os
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.swarm_based.WOA import OriginalWOA
from mealpy.swarm_based.ZOA import OriginalZOA
from mealpy.swarm_based.GWO import OriginalGWO
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.ES import CMA_ES
from algorithms.woazoa import WOAZOA
from algorithms.gwozoa import GWOZOA    
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
    "gwozoa": GWOZOA,
}

CLASSICAL_ALGO = {
    "otsu": otsu_thresholding,
    "kmeans": kmeans_thresholding
}

DEFAULT_POP_SIZE = 50
DEFAULT_EPOCHS = 100

def segment_image(image, k, thresholds, mode="color"):
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
    if params is None:
        params = {}
        
    if algorithm_name in NATURE_ALGO:
        problem = SegmentationProblem(image, k, mode)
        params.setdefault("epoch", DEFAULT_EPOCHS)
        params.setdefault("pop_size", DEFAULT_POP_SIZE)
        model = NATURE_ALGO[algorithm_name](**params)
        model.solve(problem)
        thresholds = np.sort(model.g_best.solution).clip(1,254).astype(int).tolist()
        segmented_image = segment_image(image, k, thresholds, mode)
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
