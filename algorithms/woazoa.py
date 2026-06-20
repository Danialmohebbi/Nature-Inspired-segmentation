import numpy as np
from mealpy.optimizer import Optimizer

class WOAZOA(Optimizer):
    def __init__(self, epoch=100, pop_size=50, alpha=2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        
    
    def evolve(self,epoch):
        if self.pop is None:
            return
        if self.g_best is None:
            return
        best_solution = self.g_best.solution.copy()
        
        a = 2 - 2 * (epoch / self.epoch)
        
        for idx in range(self.pop_size):
            
            position = self.pop[idx].solution.copy()
            p = (1 - epoch / self.epoch) ** self.alpha
            u = np.random.rand()
            if u < p:
                """ZOA Exploration phase"""
                j, k = np.random.choice(range(self.pop_size), 2, replace=False)
                x_j = self.pop[j].solution
                x_k = self.pop[k].solution
                r_1, r_2 = np.random.rand(), np.random.rand()
                new_position = position + r_1 * (x_j - x_k) + r_2 * (best_solution - position)
            else:
                """WOA Exploitation phase"""
                r_1, r_2 = np.random.rand(), np.random.rand()
                A = 2 * a * r_1 - a
                C = 2 * r_2
                q = np.random.rand()
                if q < 0.5:
                    new_position = best_solution - A * np.abs(C * best_solution - position)
                else:
                    b = 1
                    l = np.random.uniform(-1, 1)
                    new_position = np.abs(best_solution - position) * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
            
            new_position = self.correct_solution(new_position)
            new_agent = self.generate_agent(new_position)
            minmax = getattr(self.problem, "minmax", None) or "min"
            if self.compare_target(new_agent.target, self.pop[idx].target, minmax):
                self.pop[idx] = new_agent