
import numpy as np
from mealpy.optimizer import Optimizer

class GWOZOA(Optimizer):
    def __init__(self, epoch=100, pop_size=50, exploit_scale=0.1,strategy_switch_prob=0.01, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.exploit_scale = exploit_scale
        self.strategy_switch_prob = strategy_switch_prob
    
    def evolve(self, epoch):
        sorted_pop = sorted(self.pop, key=lambda agent: agent.target.fitness)
        alpha, beta, delta = sorted_pop[0], sorted_pop[1], sorted_pop[2]
        
        lb = np.array(self.problem.lb)
        ub = np.array(self.problem.ub)
        a = 2 - 2 * (epoch / self.epoch)
        
        for idx in range(self.pop_size):
            position = self.pop[idx].solution.copy()
            
            r_1, r_2 = np.random.rand(), np.random.rand()
            A = 2 * a * r_1 - a
            C = 2 * r_2
            
            D1 =  np.abs(C * alpha.solution - position)
            D2 = np.abs(C * beta.solution - position)
            D3 = np.abs(C * delta.solution - position)
            X1 = alpha.solution - A * D1
            X2 = beta.solution - A * D2
            X3 = delta.solution - A * D3
            X_temp = (X1 + X2 + X3) / 3
            
            if np.random.rand() < self.strategy_switch_prob:
                R = self.exploit_scale
                X_final = X_temp + R * (ub - lb) * 2 * (np.random.rand(self.probblem.n_dims) - 1)
            else:
                X_final = X_temp
                
            X_final = self.correct_solution(X_final)
            new_agent = self.generate_agent(X_final)
            if self.compare_target(new_agent.target, self.pop[idx].target,self.problem.minmax):
                self.pop[idx] = new_agent
                