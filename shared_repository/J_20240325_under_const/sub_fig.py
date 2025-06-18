import os, model, load_files, opt_problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split_yield(param_s, _data, get_each_yield_cal, conversion=False):
    z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(_data, M1, M2, conversion=conversion)
    yield_calc = get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, param_s, T)
    return yield_calc, yield_act

def draw_yield_fig(pre, act, title, filename):
    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(act, color='tab:orange', label='actual')
    ax.plot(pre, color='tab:blue', label='calclate')
    # plt.xlim([200, 300])
    plt.ylim([0.5, 0.75])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)

s_opt = np.array(pd.read_csv(os.getcwd() + '/files/M400_s_opt_re.csv').iloc[:, 1])

df, fitting_df, fitting_data = load_files.load_files()
unit_conversion = True
M1 = 254.24
M2 = 400 # 200~500 g/mol
# 収率算出
training_data = fitting_df[(4 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 9)]
verification_data = fitting_df[(10 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 11)]

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
result_df.to_csv(f'./files/M{M2}_yield_mean_re.csv')

draw_yield_fig(yield_tra, yield_tra_act, f'yield (training) (M2={M2}, conversion={unit_conversion})', f'./fig/training_M{M2}_re.png')
draw_yield_fig(yield_ver, yield_ver_act, f'yield (prediction) (M2={M2}, conversion={unit_conversion})', f'./fig/prediction_M{M2}_re.png')

print('save figure and file results')