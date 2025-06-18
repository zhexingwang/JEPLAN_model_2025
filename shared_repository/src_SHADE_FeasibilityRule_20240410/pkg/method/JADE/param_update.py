#coding: utf-8
import numpy as np
import math

class param_update_class:
    def __init__(self, m):
        self.m = m
        self.mu_F = 0.9
        self.mu_C = 0.5
        self.sigma_F = 0.1
        self.sigma_C = 0.1
        self.c = 0.1

    def rand_cauchy(self, mu, sigma, array):
        m = len(array)
        onethird_m = m//3
        param = np.zeros(self.m)
        random_onethird_idx = np.random.choice(array, size=onethird_m, replace=False).tolist()
        idx_ = [i for i in range(0, m) if i in random_onethird_idx]
        idx_not = list(set(range(0, m)) - set(idx_))
        param[idx_] = 1.2 * np.random.rand(len(idx_))
        param[idx_not] = np.random.normal(mu, sigma, len(idx_not))
        return param

    def update_param(self, s_F, s_C):
        def _lehmar_mean(array):
            return np.sum(np.power(array, 2))/np.sum(array)

        def _get_F(mu_F, sigma_F):
            F = self.rand_cauchy(mu_F, sigma_F, np.arange(0, self.m))
            F = np.clip(F, 0.01, 1)
            return F

        def _get_C(mu_C, sigma_C):
            C = np.random.normal(mu_C, sigma_C, self.m)
            C = np.clip(C, 0.01, 1)
            return C

        # update mu_F, mu_C
        if len(s_F) > 0:
            self.mu_F = (1-self.c)*self.mu_F + self.c*_lehmar_mean(s_F)
        if len(s_C) > 0:
            self.mu_C = (1-self.c)*self.mu_C + self.c*np.mean(s_C)

        # generate F, C
        F = _get_F(self.mu_F, self.sigma_F)
        C = _get_C(self.mu_C, self.sigma_C)
        return F, C