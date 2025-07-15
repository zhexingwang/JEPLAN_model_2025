#coding: utf-8
import numpy as np

class crossover_class:
    def __init__(self, C, x1_, x2_):
        self.x1 = x1_.copy()
        self.x2 = x2_.copy()
        (self.m, self.N) = x1_.shape
        self.C = C*np.ones(self.m)

    def binomial(self):
        # theta: (m, N)
        theta = np.random.rand(self.m, self.N)
        lambda_ = np.random.randint(0, self.N, size=self.m)
        # binomial crossover
        child = np.copy(self.x1)
        for i in range(0, self.m):
            nidx = np.where(theta[i, :] <= self.C[i])
            child[i, nidx] = self.x2[i, nidx]
            child[i, lambda_[i]] = self.x2[i, lambda_[i]]
        return child