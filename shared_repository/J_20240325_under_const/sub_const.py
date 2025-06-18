import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
# import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import time
# import load_files, model, opt_problem
import load_files
import opt_problem_jDE as opt_problem
import model_jDE as model
import pygmo as pg
import random

def split_yield(param_s, _data, get_each_yield_cal, M1, M2, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    yield_calc = get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, param_s, T)
    return yield_calc, yield_act
def const_(_data, get_each_const, M1, M2, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    const = get_each_const(z1_mol, z1, p_total, M1, M2, F_in, T)
    return np.concatenate(const)

def draw_evol_fig(progress_df, cal_time, filename):
    plt.plot(progress_df.iloc[:, 0], progress_df.iloc[:, 2])
    plt.xlabel('Iteration')
    plt.ylabel('Evaluation Value')
    plt.grid(True)
    plt.text(0.91, 0.91, 'time = %.2f sec = %.2f min' % (cal_time, cal_time/60), va='top', ha='right', transform=plt.gca().transAxes)
    plt.savefig(filename)

def draw_yield_fig(pre, act, title, filename):
    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(act, color='tab:orange', label='actual')
    ax.plot(pre, color='tab:blue', label='calclate')
    # plt.xlim([200, 300])
    plt.ylim([0.50, 0.75])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)

# 最適化のログ取得
progress = []
class Callback(object):
    def __init__(self, Obj):
        self.nit = 0
        self.Obj = Obj

    def __call__(self, xk, convergence):
        self.nit += 1
        #np.testing.assert_equal(xk, self.Obj.best_x)
        print(self.nit, self.Obj.best_s, self.Obj.minf)
        progress.append((self.nit, self.Obj.best_s, self.Obj.minf))

class fit_s_:
    def __init__(self, pop, algo):
        self.pop=pop
        self.f=pop.get_f()[pop.best_idx()]
        self.x=pop.get_x()[pop.best_idx()] 
        self.constr_violation=None
        uda = algo.extract(pg.sade)
        self.log=uda.get_log()
        i = 1
        for n in self.log:
            progress.append((i, self.x, n[2]))
            i += 1

    def __str__(self):
        strlog = str(self.pop) + '\n'
        strlog += 'Gen,Fevals,Best,F,CR,dx,df:\n'
        for s in self.log:
            strlog += str(s) + '\n'
        return strlog

# 最適化アルゴリズムの設定
class optimization:
    def __init__(self, fitting_data, M1, M2, conversion=False):
        self.conversion = conversion
        z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(fitting_data, M1, M2, conversion=self.conversion)
        self.bounds = opt_problem.def_bounds()
        self.s_ini = opt_problem.initialize_s()
        self.obj = opt_problem.obj(z1_mol, z1, p_total, M1, M2, F_in, T, yield_act, model.get_each_yield_cal, model.abs_diff)
        self.Const = model.cst(fitting_data, M1, M2, conversion=False)
        self.const = self.Const.get_each_const(fitting_data, M1, M2)
        self.const2 = self.Const.get_each_const_jDE(fitting_data, M1, M2)
        prob2=pg.problem(model.my_constrained_udp(self.obj, self.const2))
        # print(prob2)
        self.prob=pg.unconstrain(prob=prob2, method='death penalty', weights=[])
    # DEでない方は未完成
    def opt(self, _maxiter=5, _polish=False, solver='DE'):
        if solver=='DE':
            C = Callback(self.obj)
            fit_s = differential_evolution(self.obj, self.bounds, x0=self.s_ini, disp=True, updating='deferred', maxiter=_maxiter, polish=_polish, tol=0.0001, callback=C, constraints=self.const, seed=300)
        elif solver=='L-BFGS-B':
            jac = jacobian(self.obj.E)
            hess = jacobian(jac)
            fit_s = minimize(method='L-BFGS-B', fun=self.obj.E, x0=self.s_ini, jac=jac, bounds=self.bounds, options={'nit': 5})
        elif solver=='jDE':
            algo=pg.algorithm(pg.sade(gen=_maxiter, variant=7, variant_adptv=1, ftol=1e-6, xtol=1e-6, memory=False, seed=0))
            algo.set_verbosity(1)  # Set the verbosity of logs and screen output.
            pop=pg.population(prob=self.prob, size=100)  # size: the number of individuals
            # for i in range(100):    # give initial population
            #     rnd = np.array([random.random() for j in range(len(self.s_ini))])
            #     s_ini_np=np.array(self.s_ini)
            #     pop.set_x(i, (s_ini_np + s_ini_np * rnd * 0.01).tolist())
            #     # pop.set_x(i, self.s_ini)
            # init_x = pop.get_x()
            pop=algo.evolve(pop)
            fit_s=fit_s_(pop, algo)
        return fit_s

'''
ここで変更するもの
・M2の値
・solver='DE': maxiterの値, polishの有無
・unit_conversionの有無
'''
M1 = 254.24
M2 = 400 # 200~500 g/mol
maxiter = 50
# polish = True
polish = False
unit_conversion = True
# unit_conversion = False

print('import files')
df, fitting_df, fitting_data = load_files.load_files()

print('set optimization problem')
opt = optimization(fitting_data, M1, M2, conversion=unit_conversion)

'''
test
'''
# const = const_(fitting_data, model.get_each_const, conversion=unit_conversion)

print('start optimization')
start_time = time.time()
fit_s = opt.opt(_maxiter=maxiter, _polish=polish, solver='jDE')
end_time = time.time()
cal_time = end_time - start_time
print("violation value=" + str(fit_s.constr_violation))
# print(fit_s)
print('optimization is terminated')
print('time = %f sec = %f min' % (cal_time, cal_time/60))

f = open(f'./files/M{M2}_optimization_logs.txt', 'w')
for n in progress:
    f.write(str(n) + '\n')
f.close()

progress_df = pd.DataFrame(progress)
draw_evol_fig(progress_df, cal_time, f'./fig/evaluation_plot.png')

f = open(f'./files/M{M2}_optimization_results.txt', 'w')
f.write(str(fit_s) + '\n')
f.write(str(cal_time) + ' [sec] = ' + str(cal_time/60) + '[min] \n')
f.close()

print('results')
s_opt = fit_s.x
s_opt_df = pd.DataFrame(s_opt, index=['A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'alpha', 'd_g12', 'd_g21'])
s_opt_df.to_csv(f'./files/M{M2}_s_opt.csv')

# 収率算出
training_data = fitting_df[(4 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 9)]
verification_data = fitting_df[(10 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 11)]
# verification_data = fitting_df[fitting_df['Batch Number'] == 7]

'''
constのテスト
'''
# const = const_(fitting_data, model_copy.get_each_const, conversion=unit_conversion)
# const = const_test(fitting_data, model_copy.get_each_const_, conversion=unit_conversion)

yield_tra, yield_tra_act = split_yield(s_opt, training_data, model.get_each_yield_cal, M1, M2, conversion=unit_conversion)
yield_ver, yield_ver_act = split_yield(s_opt, verification_data, model.get_each_yield_cal, M1, M2, conversion=unit_conversion)

result_df = pd.DataFrame(np.zeros((3, 2)), index=['mean actual value', 'mean predicted value', 'RMSE'], columns=['training', 'prediction'])
result_df.at['mean actual value', 'training'] = yield_tra_act.mean()
result_df.at['mean predicted value', 'training'] = yield_tra.mean()
result_df.at['RMSE', 'training'] = np.sqrt(model.abs_diff(yield_tra_act, yield_tra).mean())
result_df.at['mean actual value', 'prediction'] = yield_ver_act.mean()
result_df.at['mean predicted value', 'prediction'] = yield_ver.mean()
result_df.at['RMSE', 'prediction'] = np.sqrt(model.abs_diff(yield_ver_act, yield_ver).mean())

print(result_df)
result_df.to_csv(f'./files/M{M2}_yield_mean.csv')

draw_yield_fig(yield_tra, yield_tra_act, f'yield (training) (M2={M2}, conversion={unit_conversion})', f'./fig/training_M{M2}.png')
draw_yield_fig(yield_ver, yield_ver_act, f'yield (prediction) (M2={M2}, conversion={unit_conversion})', f'./fig/prediction_M{M2}.png')

print('save figure and file results')