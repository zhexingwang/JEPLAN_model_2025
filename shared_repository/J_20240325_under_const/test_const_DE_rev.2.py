import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, LinearConstraint, NonlinearConstraint
from autograd import grad, jacobian
import time

class optimization:
    def __init__(self, f):
        #X = opt_problem.initialize_X()
        self.bound = ([-10, 10],[-10, 10],[-10, 10])
        self.X = [(b[1]-b[0])*np.random.uniform()+b[0] for b in self.bound]
        self.obj = f

    def opt(self, solver='DE', const=[]):
        if solver=='DE':
            fit_x = differential_evolution(self.obj, self.bound, x0=self.X, disp=True, updating='deferred', maxiter=500, polish=False, tol=pow(10, -8))
        elif solver=='const_DE':
            fit_x = differential_evolution(self.obj, self.bound, x0=self.X, constraints=const, disp=True, updating='deferred', maxiter=500, polish=False, tol=pow(10, -8))
        elif solver=='L-BFGS-B':
            jac = jacobian(self.obj)
            hess = jacobian(jac)
            fit_x = minimize(method='L-BFGS-B', fun=self.obj, x0=self.X, bounds=self.bound, jac=jac, hess=hess, options={'nit': 100})
        return fit_x

class Constraints:
    def __init__(self):
        self.g_num = len(self.constraints(np.array([0.0, 0.0, 0.0])))
        self.lowerupper = [[0, 0] for i in range(self.g_num)]
        # lower
        self.lowerupper[0][0] = 0.1
        self.lowerupper[0][1] = -0.1
        # upper
        self.lowerupper[1][0] = 2
        self.lowerupper[1][1] = 0.1

    def get_intermedeate(self, c):
        x, y, z = c
        xx = z**2
        yy = x - y
        return xx, yy
    
    def constraints(self, c):
        xx, yy = self.get_intermedeate(c)
        each_const = []
        each_const.append(xx)
        each_const.append(yy)
        return tuple(each_const)
    
    def get_const(self, const_class):
        if const_class == 'Linear':
            # -inf <= A @ x <= 1.9
            # -0.1 <= x-y <= 0.1
            return LinearConstraint([[1, -1, 0]], -0.1, 0.1)
        elif const_class == 'Nonlinear':
            # 0.1 <= z**2 <= 2 :  ?
            return NonlinearConstraint(self.constraints, self.lowerupper[0], self.lowerupper[1], keep_feasible=True)

def get_each_constraint(c):
    def _get_constraint_minmax(c):
        x, y, z = c
        xx = z**2
        yy = x - y
        const = []
        # g_equgl, lower, upper
        # 0.1 <= z**2 <= 2
        const.append([xx, 0.1, 2])
        # -0.1 <= x-y <= 0.1
        const.append([yy, -0.1, 0.1])
        return const

    def _get_oneside_vio(g_equal, lower, upper):
        return np.where((lower + upper)/2>=g_equal, lower - g_equal, g_equal - upper)

    const = _get_constraint_minmax(c)
    # gj(x)
    each_const = np.zeros(len(const))
    for i, const_ele in enumerate(const):
        # lower <= xx <= upper => g = xx - upper, lower - xx
        each_const[i] = _get_oneside_vio(const_ele[0], const_ele[1], const_ele[2])
    return each_const

def get_each_violation(c):
    each_const = get_each_constraint(c)
    # max(gj(x), 0)
    return np.max([np.zeros(len(each_const)), each_const], axis=0)

def get_obj(c):
    # optimization variables
    x, y, z = c
    return x**2 + (y**2 + z**2 + 1) * (2 + np.sin(y))

def get_lagrangian(c):
    # penalty coefficient
    coef = pow(10, 2)
    each_vio = get_each_violation(c)
    return get_obj(c) + coef * np.array(each_vio).sum()




# test: (1,1,1) => each_vio
each_vio = get_each_violation(np.array([1.0, 1.0, 1.0]))

# method: penalty, constraint_option
# method = 'penalty'
method = 'constraint_option'
# Nonlinear or Linear
const_class = 'Nonlinear'
print(const_class)

print('set optimization problem: ' + method)

if method == 'penalty':
    opt = optimization(get_lagrangian)
elif method == 'constraint_option':
    opt = optimization(get_obj)
    Const = Constraints()
    lc = Const.get_const(const_class)

print('start optimization: ' + method)

start_time = time.time()
# solver: 'L-BFGS-B', 'DE'
if method == 'penalty':
    fit_x = opt.opt(solver='DE')
elif method == 'constraint_option':
    fit_x = opt.opt(solver='const_DE', const=lc)
end_time = time.time()
cal_time = end_time - start_time
print(fit_x)

# test: optimized_solution => each_vio
# each_vio = 0 => feasible
each_vio = get_each_violation(fit_x.x)
print('each_vio = ', each_vio)

print('optimization is terminated: ' + method)
print('time = %f sec = %f min' % (cal_time, cal_time/60))
