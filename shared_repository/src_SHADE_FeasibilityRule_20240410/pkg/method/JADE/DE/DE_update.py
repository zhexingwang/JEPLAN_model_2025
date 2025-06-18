#coding: utf-8
import numpy as np
from .crossover import crossover_class
from .mutation import mutation_class
from ....subpkg.get_eval import Superior

class x_update_class:
    def __init__(self, m, N):
        self.m = m
        self.N = N
        self.mutation_type = 'CP1'
        self.crossover_type = 'binomial'
        self.rho = 0.05

    def neighbor_gene(self, prob, x_, obj_, mod):
  	    # x_: (m, N)
	    # obj_: (m,)
        num_diff = 1
        
        #mutation
        mutate = mutation_class(self.F, x_)
        if self.mutation_type == 'rand1':
            u = mutate.rand(num_diff=num_diff)
        elif self.mutation_type == 'CP1':
            num_better = np.max([self.m*self.rho, 2]).astype(int)
            idx_rank = Superior().get_rank(obj_, prob.eval_type)
            better_idx = idx_rank[:num_better]
            u = mutate.CP(better_idx, num_diff=num_diff)

        #crossover
        xover = crossover_class(self.C, x_, u)
        if self.crossover_type == 'binomial':
            u = xover.binomial()
        
        #modify
        mod_x = mod.modified_x(u)
        
        return mod_x

    def selection(self, prob, x__, obj__, x_nei__, obj_nei__):
        update_idx_ = Superior().get_sup_idx_array(obj__, obj_nei__, prob.eval_type)
        if len(update_idx_) > 0:
            x__[update_idx_, :] = x_nei__[update_idx_, :].copy()
            obj__[update_idx_, :] = obj_nei__[update_idx_, :].copy()
        return x__, obj__, update_idx_

    def DE_update(self, prob, x, obj, mod):
        x_nei = self.neighbor_gene(prob, x, obj, mod)
        (obj_nei, each_vio_nei) = Superior().eval_scaler(prob, x_nei)
        (x_, obj_, update_idx_) = self.selection(prob, x, obj, x_nei, obj_nei)
        self.update_idx_ = update_idx_
        return x_, obj_