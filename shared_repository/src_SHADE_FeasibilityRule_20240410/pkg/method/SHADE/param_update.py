#coding: utf-8
import numpy as np

class param_update_class:
    def __init__(self, m):
        self.m = m
        self.sigma_F = 0.1
        self.sigma_C = 0.1
        self.memory_size = 10
        self.set_F_C = np.ones((self.memory_size, 2))*0.5

    def rand_cauchy(self, mu, sigma, array):
        m = len(array)
        onethird_m = m//3
        param = np.zeros(self.m)
        random_onethird_idx = np.random.choice(array, size=onethird_m, replace=False).tolist()
        idx_ = [i for i in range(0, m) if i in random_onethird_idx]
        idx_not = list(set(range(0, m)) - set(idx_))
        param[idx_] = 1.2 * np.random.rand(len(idx_))
        for i in idx_not:
            param[i] = np.random.normal(mu[i], sigma)
        return param

    def rand_gauss(self, mu, sigma):
        param = np.zeros(self.m)
        for (i, mu_i) in enumerate(mu):
            param[i] = np.random.normal(mu_i, sigma)
        return param

    def update_param(self, s_F, s_C, iter):
        def _lehmar_mean(array):
            return np.sum(np.power(array, 2))/np.sum(array)

        def _get_F(mu_F, sigma_F):
            F = self.rand_cauchy(mu_F, sigma_F, np.arange(0, self.m))
            F = np.clip(F, 0.01, 1)
            return F

        def _get_C(mu_C, sigma_C):
            C = self.rand_gauss(mu_C, sigma_C)
            C = np.clip(C, 0.01, 1)
            return C

        # update memory
        if len(s_F) > 0 or len(s_C) > 0:
            idx = np.random.randint(0,self.memory_size,2)
            if len(s_F) > 0:
                self.set_F_C[idx, 0] = _lehmar_mean(s_F)
            if len(s_C) > 0:
                self.set_F_C[idx, 1] = _lehmar_mean(s_C)

        # get mu_F, mu_C
        if iter < self.memory_size:
            idx_F = [iter] * self.m
            idx_C = [iter] * self.m
        else:
            r_idx = np.random.choice(np.arange(0, self.memory_size), (self.m,2))
            idx_F = r_idx[:,0]
            idx_C = r_idx[:,1]
        mu_F = self.set_F_C[idx_F, 0]
        mu_C = self.set_F_C[idx_C, 1]

        # generate F, C
        F = _get_F(mu_F, self.sigma_F)
        C = _get_C(mu_C, self.sigma_C)
        return F, C