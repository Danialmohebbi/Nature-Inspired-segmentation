from mealpy import GA,ES
import numpy as np

def ga(image, k, mode="color", epoch=100, pop_size=50, crossover_rate=0.8, mutation_rate=0.2):
    problem = SegmentationProblem(image, k, mode)
    model= GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=crossover_rate, pm=mutation_rate)
    model.solve(problem)
    
    thresholds = np.sort(model.g_best.solution).clip(1,254).astype(np.uint8).tolist()
    
    return thresholds, model.g_best.target.fitness, model.history.list_global_best_fit

def cmaes(image, k, mode="color", epoch=100, pop_size=50):
    problem = SegmentationProblem(image, k, mode)
    model = ES.CMA_ES(epoch=epoch, pop_size=pop_size)
    model.solve(problem)
    
    thresholds = np.sort(model.g_best.solution).clip(1,254).astype(np.uint8).tolist()
    
    return thresholds, model.g_best.target.fitness, model.history.list_global_best_fit