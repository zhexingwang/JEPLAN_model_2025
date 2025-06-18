import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint

def antoine(T, A, B, C):
    p = 10**(A - (B / (T + C)))
    return p

class cst:
    def __init__(self, conversion=False):
        self.conversion = conversion
        self.T_array = 1.5*np.arange(20)
        self.g_num = 5
        self.lowerupper = [[0, 0] for i in range(self.g_num)]
        self.lowerupper[0][0] = 0
        self.lowerupper[0][1] = 20
        self.lowerupper[1][0] = 0
        self.lowerupper[1][1] = 20

    def p1(self, x):
        return self.T - x[1]

    def x1(self, x):
        return self.T - x[0]

    def const_cal_(self):
        return [NonlinearConstraint(self.p1, self.lowerupper[0][0], self.lowerupper[0][1]),
                NonlinearConstraint(self.x1, self.lowerupper[1][0], self.lowerupper[1][1]),
                ]

    def get_each_const(self):
        constr = []
        for i in range(len(self.T_array)):
            self.T = self.T_array[i]
            constr.append(self.const_cal_())
        return np.concatenate(np.array(constr))

def obj(x):
    return np.power(x, 2).mean()

# L<=x<=U => v = L - x, v = x - U

const = cst()
c = const.get_each_const()
bounds = [(-5, 5), (-5, 5), (-5, 5)]
s_ini = np.random.uniform(-5, 5, 3)
fit_s = differential_evolution(obj, bounds, x0=s_ini, disp=True, updating='deferred', maxiter=2, polish=False, tol=0.0001, constraints=c)

print(fit_s.constr_violation)

# SVNのテスト