import numpy as np
import matplotlib.pyplot as plt
from model_JEPLAN2 import get_each_yield_cal, get_each_yield_cal_all
import load_files_JEPLAN
import opt_problem_JEPLAN2 as opt_problem

def split_yield(param_s, _data, get_each_yield_cal, M1, M2, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    yield_calc = get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, param_s, T)
    return yield_calc, yield_act

def split_yield_tra(param_s, _data, get_each_yield_cal, M1, M2, T_opt, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    yield_calc = get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, param_s, T_opt)
    return yield_calc, yield_act

def draw_yield_fig(pre, act, title, filename):
    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(act, color='tab:orange', label='actual')
    ax.plot(pre, color='tab:blue', label='calclate')
    # plt.xlim([200, 300])
    # plt.ylim([0, 1])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)

M1 = 254.24
M2 = 400 # 200~500 g/mol
unit_conversion = True

print('import files')
df, fitting_df, fitting_data = load_files_JEPLAN.load_files()

# f = f'./output/pro6/file/x_gbest_final.csv'
# a_opt_np = np.loadtxt(f, delimiter=',', skiprows=1, usecols=1)
a_opt_np = np.ones(9)
a_opt_np[0] = 5.355448416   # A1
a_opt_np[1] = 1567.272845   # B1
a_opt_np[2] = 230.1241588   # C1
a_opt_np[3] = 5.00416381    # A2
a_opt_np[4] = 1998.351011   # B2
a_opt_np[5] = 225.3158741   # C2
a_opt_np[6] = 0.33915189    # alpha
a_opt_np[7] = -4949.805319  # d_g12
a_opt_np[8] = -4678.417759  # d_g21
s_opt = a_opt_np

f = f'./output/pro6/file/x_gbest_final.csv'
T_opt = np.loadtxt(f, delimiter=',', skiprows=1, usecols=1)

# 収率算出
training_data = fitting_df[(4 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 9)]
# training_data = fitting_df[(12 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 12)]
verification_data = fitting_df[(10 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 11)]

yield_tra, yield_tra_act = split_yield_tra(s_opt, training_data, get_each_yield_cal, M1, M2, T_opt, conversion=unit_conversion)
yield_ver, yield_ver_act = split_yield(s_opt, verification_data, get_each_yield_cal, M1, M2, conversion=unit_conversion)

draw_yield_fig(yield_tra, yield_tra_act, f'yield (training) (M2={M2}, conversion={unit_conversion})', f'./output/pro6/fig/training_M{M2}.png')
draw_yield_fig(yield_ver, yield_ver_act, f'yield (prediction) (M2={M2}, conversion={unit_conversion})', f'./output/pro6/fig/prediction_M{M2}.png')

f = open(f'./output/pro6/file/yield_training_act_cal.csv', 'w')
for n1, n2 in zip(yield_tra_act, yield_tra):
    f.write(str(n1) + ',' + str(n2) + '\n')
f.close()

print('training')
rmse = np.sqrt(np.mean((yield_tra - yield_tra_act) ** 2))
print('RMSE: ' + str(rmse))

yield_tra_act_mean = np.mean(yield_tra_act)
print('actual_mean: ' + str(yield_tra_act_mean))

yield_tra_mean = np.mean(yield_tra)
print('calc_mean: ' + str(yield_tra_mean))

print('prediction')
rmse = np.sqrt(np.mean((yield_ver - yield_ver_act) ** 2))
print('RMSE: ' + str(rmse))

yield_ver_act_mean = np.mean(yield_ver_act)
print('actual_mean: ' + str(yield_ver_act_mean))

yield_ver_mean = np.mean(yield_ver)
print('calc_mean: ' + str(yield_ver_mean))


z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(training_data, M1, M2, conversion=unit_conversion)
yield_calc = get_each_yield_cal_all(z1_mol, z1, p_total, M1, M2, F_in, s_opt, T_opt)

np.savetxt(f'./output/pro6/file/result_intermediate.csv', np.append(yield_calc, F_in.reshape(F_in.shape[0], 1), 1), delimiter=',')