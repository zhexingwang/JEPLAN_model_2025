#coding: utf-8
import numpy as np

class mutation_class:
    def __init__(self, F, x_):
        self.x = x_.copy()
        (self.m, self.N) = x_.shape
        self.F = F*np.ones(self.m)

    def diff_one(self, r):
        # r: (m, 2)
        u = self.x[r[:, 0],:] - self.x[r[:, 1],:]
        return u

    def rand(self, num_diff=1):
        child = np.zeros((self.m, self.N))
        if num_diff == 1:
            r = np.zeros((self.m, 3)).astype(int)
            idx_ori = np.arange(self.m)
            for i in range(0, self.m):
                idx = np.delete(idx_ori, np.where(idx_ori == i))
                r[i, :] = np.random.choice(idx, size=3, replace=False)
            y = self.x[r[:, 0],:]
            u = self.diff_one(r[:, [1,2]])
        else:
            u = np.zeros((self.m, self.N))
        for i in range(0, self.m):
            child[i, :] = y[i, :] + self.F[i] * u[i, :]
        return child

    def CP(self, idx_rank, num_better, num_diff=1):
        # obj: (m,), rank
        child = np.zeros((self.m, self.N))
        y = np.zeros((self.m, self.N))
        if len(num_better) > 0:
            p = np.zeros(self.m).astype(int)
            for (i, idx) in enumerate(num_better):
                p[i] = np.random.choice(idx_rank[:idx], replace=True)
        else:
            p = np.random.choice(idx_rank[:num_better], size=self.m, replace=True)
        if num_diff == 1:
            r = np.zeros((self.m, 2)).astype(int)
            idx_ori = np.arange(self.m)
            for i in range(0, self.m):
                idx = np.delete(idx_ori, np.where(idx_ori == i))
                r[i, :] = np.random.choice(idx, size=2, replace=False)
                y[i, :] = self.x[i, :] + self.F[i]*(self.x[p[i],:] - self.x[i,:])
            u = self.diff_one(r)
        else:
            u = np.zeros((self.m, self.N))        
        # (1, m) * (m, N)
        for i in range(0, self.m):
            child[i, :] = y[i, :] + self.F[i] * u[i, :]
        return child