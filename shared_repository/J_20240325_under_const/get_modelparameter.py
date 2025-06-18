import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
# import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt
import time
import load_files, model, opt_problem

def split_yield(param_s, _data, get_each_yield_cal, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    yield_calc = get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, param_s, T)
    return yield_calc, yield_act

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

# 最適化アルゴリズムの設定
class optimization:
    def __init__(self, fitting_data, M1, M2, conversion=False):
        self.conversion = conversion
        z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(fitting_data, M1, M2, conversion=self.conversion)
        self.bounds = opt_problem.def_bounds()
        self.s_ini = opt_problem.initialize_s()
        self.obj = opt_problem.obj(z1_mol, z1, p_total, M1, M2, F_in, T, yield_act, model.get_each_yield_cal, model.abs_diff)
    # DEでない方は未完成
    def opt(self, _maxiter=5, _polish=False, solver='DE'):
        if solver=='DE':
            C = Callback(self.obj)
            fit_s = differential_evolution(self.obj, self.bounds, x0=self.s_ini, disp=True, updating='deferred', maxiter=_maxiter, polish=_polish, tol=0.0001, callback=C)
        elif solver=='L-BFGS-B':
            jac = jacobian(self.obj.E)
            hess = jacobian(jac)
            fit_s = minimize(method='L-BFGS-B', fun=self.obj.E, x0=self.s_ini, jac=jac, bounds=self.bounds, options={'nit': 5})
        return fit_s

'''
ここで変更するもの
・M2の値
・solver='DE': maxiterの値, polishの有無
・unit_conversionの有無
'''
M1 = 254.24
M2 = 400 # 200~500 g/mol
maxiter = 5
polish = True
# polish = False
unit_conversion = True
# unit_conversion = False

print('import files')
df, fitting_df, fitting_data = load_files.load_files()

print('set optimization problem')
opt = optimization(fitting_data, M1, M2, conversion=unit_conversion)

print('start optimization')
start_time = time.time()
fit_s = opt.opt(_maxiter=maxiter, _polish=polish, solver='DE')
end_time = time.time()
cal_time = end_time - start_time
print(fit_s)
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
# verification_data = fitting_df[fitting_df['Batch Number'] == 4]
yield_tra, yield_tra_act = split_yield(s_opt, training_data, model.get_each_yield_cal, conversion=unit_conversion)
yield_ver, yield_ver_act = split_yield(s_opt, verification_data, model.get_each_yield_cal, conversion=unit_conversion)

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