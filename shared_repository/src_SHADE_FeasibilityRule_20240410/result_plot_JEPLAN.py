import numpy as np
import matplotlib.pyplot as plt
from model_JEPLAN import get_each_yield_cal, get_each_yield_cal_all
import load_files_JEPLAN
import opt_problem_JEPLAN as opt_problem

def split_yield(param_s, _data, get_each_yield_cal, M1, M2, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    yield_calc = get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, param_s, T)
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

f = f'./output/pro5/file/x_gbest_final.csv'
a_opt_np = np.loadtxt(f, delimiter=',', skiprows=1, usecols=1)
s_opt = a_opt_np

# 収率算出
training_data = fitting_df[(4 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 9)]
# training_data = fitting_df[(12 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 12)]
verification_data = fitting_df[(10 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 11)]

yield_tra, yield_tra_act = split_yield(s_opt, training_data, get_each_yield_cal, M1, M2, conversion=unit_conversion)
yield_ver, yield_ver_act = split_yield(s_opt, verification_data, get_each_yield_cal, M1, M2, conversion=unit_conversion)

draw_yield_fig(yield_tra, yield_tra_act, f'yield (training) (M2={M2}, conversion={unit_conversion})', f'./output/pro5/fig/training_M{M2}.png')
draw_yield_fig(yield_ver, yield_ver_act, f'yield (prediction) (M2={M2}, conversion={unit_conversion})', f'./output/pro5/fig/prediction_M{M2}.png')

f = open(f'./output/pro5/file/yield_training_act_cal.csv', 'w')
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
yield_calc = get_each_yield_cal_all(z1_mol, z1, p_total, M1, M2, F_in, s_opt, T)

np.savetxt(f'./output/pro5/file/result_intermediate.csv', np.append(yield_calc, F_in.reshape(F_in.shape[0], 1), 1), delimiter=',')