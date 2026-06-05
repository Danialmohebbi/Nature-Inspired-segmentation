import numpy as np
from mealpy import FloatVar, Problem
import sys
import os
from fitness.kapur import kapur_entropy_fitness

class SegmentationProblem(Problem):
    def __init__(self, image, k, mode="color", **kwargs):
        self.image = image
        self.k = k
        self.mode = mode
        
        thresholds_count = k - 1 if mode=="gray" else (3 * (k - 1))
        bounds = FloatVar(lb=[1] * thresholds_count, ub=[254] * thresholds_count) 
        
        super().__init__(minmax="min", bounds=bounds, **kwargs)


    def obj_func(self, candidate_solution):
        thresholds = np.sort(candidate_solution).clip(1,254).astype(int).tolist()
        return kapur_entropy_fitness(self.image, thresholds, self.mode)