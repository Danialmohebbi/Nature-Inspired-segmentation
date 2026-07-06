import numpy as np
from mealpy import FloatVar, Problem
import sys
import os
from fitness.kapur import kapur_entropy_fitness
from fitness.kapur_spatial import kapur_spatial_fitness
from fitness.kapur_ssim import kapur_ssim_fitness
FITNESS_FUNCTIONS = {
    "kapur" : kapur_entropy_fitness,
    "kapur_spatial": kapur_spatial_fitness,
    "kapur_ssim": kapur_ssim_fitness 
}


class SegmentationProblem(Problem):
    """
    Wraps the image segmentation problem as a mealpy optimization problem.
    
    The optimizer will search for the optimal threshold values that maximizes Kapur entropy accross the image histogram.

    Args:
        image (np.ndarray): uint8 np.ndarray of shape (H,W) for gray images and (H,W,3) for color images.
        k (int): Number of segments.
        mode (str): 'gray' or 'color' image.
        fitness_fn (str): It is the objective function that will be used.
        **kwargs: additional arguments.
    """
    def __init__(self, image, k, mode="color", fitness_fn="kapur",**kwargs):
        self.image = image
        self.k = k
        self.mode = mode
        if fitness_fn not in FITNESS_FUNCTIONS:
            raise ValueError(f"Unknown fitness_fn '{fitness_fn}.'")
        self.fitness_fn_name = fitness_fn
        self.fitness_fn = FITNESS_FUNCTIONS[fitness_fn]
        thresholds_count = k - 1 if mode=="gray" else (3 * (k - 1))
        bounds = FloatVar(lb=[1] * thresholds_count, ub=[254] * thresholds_count) 
        
        super().__init__(minmax="min", bounds=bounds, **kwargs)


    def obj_func(self, x):
        """
        Objective function for mealpy which is used to evaluate a candidate solution
        Args:
            x (np.ndarray): Float array of threshold values

        Returns:
            float: Negative kapur entropy. Lower is better. Minimizing this value corresponds to maximizing kapur's entropy.
        """
        thresholds = np.sort(x).clip(1,254).astype(int).tolist()
        return self.fitness_fn(self.image, thresholds,self.k, self.mode)