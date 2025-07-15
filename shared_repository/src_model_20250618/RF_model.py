import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import random

def draw_yield_fig(pre, act, title, filename):
    fig = plt.figure(figsize=(12, 3), dpi=200)
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
    print('prediction')
    rmse = np.sqrt(np.mean((pre - act) ** 2))
    print('RMSE: ' + str(rmse))

def draw_yield_fig2(pre, pre1, act, title, filename):
    fig = plt.figure(figsize=(12, 3), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(act, color='tab:orange', label='actual')
    ax.plot(pre, color='tab:blue', label='first-principles model')
    ax.plot(pre1, color='tab:green', label='hybrid model')
    # plt.xlim([200, 300])
    # plt.ylim([0, 1])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    print('prediction')
    rmse = np.sqrt(np.mean((pre - act) ** 2))
    print('first-principles model RMSE: ' + str(rmse))
    rmse1 = np.sqrt(np.mean((pre1 - act) ** 2))
    print('hybrid model RMSE: ' + str(rmse1))

def draw_yield_fig3(pre, pre1, act, title, filename):
    fig = plt.figure(figsize=(12, 3), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(act, color='tab:orange', label='actual')
    ax.plot(pre, color='tab:blue', label='first-principles model')
    ax.plot(pre1, color='tab:green', label='RF model')
    # plt.xlim([200, 300])
    # plt.ylim([0, 1])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    print('prediction')
    rmse = np.sqrt(np.mean((pre - act) ** 2))
    print('first-principles model RMSE: ' + str(rmse))
    rmse1 = np.sqrt(np.mean((pre1 - act) ** 2))
    print('RF model RMSE: ' + str(rmse1))

def draw_yield_fig4(pre, pre1, pre2, act, title, filename):
    random.seed(42)
    np.random.seed(42)
    fig = plt.figure(figsize=(12, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(act, color='tab:orange', label='actual', linestyle='-')
    ax.plot(pre, color='tab:blue', label='first-principles model', linestyle='--')
    ax.plot(pre1, color='tab:green', label='RF model', linestyle='--')
    ax.plot(pre2, color='tab:pink', label='hybrid model', linestyle='--')
    # plt.xlim([200, 300])
    plt.ylim([0.60, 0.80])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    print('prediction')
    rmse = np.sqrt(np.mean((pre - act) ** 2))
    print('first-principles model RMSE: ' + str(rmse))
    rmse1 = np.sqrt(np.mean(pre1 - act) ** 2)
    print('RF model RMSE: ' + str(rmse1))
    rmse2 = np.sqrt(np.mean(pre2 - act) ** 2)
    print('hybrid model RMSE: ' + str(rmse2))

def mat(data_set_normal, png_name):
    # scatter, hist, plot
    plt.figure(figsize=(10, 10))
    # g = sns.PairGrid(data_set_normal.drop("TimeStamp", axis=1), diag_sharey=False)
    g = sns.PairGrid(data_set_normal, diag_sharey=False)
    g.map_diag(sns.histplot)
    g.map_upper(sns.scatterplot)
    # g.map_lower(sns.histplot)
    # g.map_lower(sns.kdeplot, fill=True, cmap='coolwarm')
    g.map_lower(sns.kdeplot, fill=True, cmap='rainbow')
    g.savefig(png_name)

# 学習データの準備
## 誤差データ
error_training_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/output/pro5/file/yield_training_act_cal.csv', names=["error_training_act", "error_training_cal"])
error_verification_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/output/pro5/file/yield_verification_act_cal.csv', names=["error_verification_act", "error_verification_cal"])
# print(error_training_data.head())
## 特徴量
EE3_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/fitting_dataset/v3_result_df_202401121430.csv',\
                        usecols=["JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV", "JEP.EU_WZIRA-EE3-201.PV", "Batch Number"])
# EE3_training_data = EE3_data[(4 <= EE3_data['Batch Number']) & (EE3_data['Batch Number'] <= 9)]
EE3_training_data = EE3_data[(4 <= EE3_data['Batch Number']) & (EE3_data['Batch Number'] <= 6)]
# EE3_verification_data = EE3_data[(10 <= EE3_data['Batch Number']) & (EE3_data['Batch Number'] <= 11)]
EE3_verification_data = EE3_data[(7 <= EE3_data['Batch Number']) & (EE3_data['Batch Number'] <= 9)]
# print(EE3_verification_data)
BD_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '/fitting_dataset/v3_result_df_202401121430.csv',\
                        usecols=["JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV", "Batch Number"])
# BD_training_data = BD_data[(4 <= BD_data['Batch Number']) & (BD_data['Batch Number'] <= 9)]
BD_training_data = BD_data[(4 <= BD_data['Batch Number']) & (BD_data['Batch Number'] <= 6)]
# BD_verification_data = BD_data[(10 <= BD_data['Batch Number']) & (BD_data['Batch Number'] <= 11)]
BD_verification_data = BD_data[(7 <= BD_data['Batch Number']) & (BD_data['Batch Number'] <= 9)]

feature_num = "BD"
# 学習
## 学習データ
# X_train = np.reshape(EE3_training_data["JEP.EU_DIRA-EE3-201.PV"]._values, [-1,1])
if feature_num == "EE3":
    X_train = EE3_training_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV", "JEP.EU_WZIRA-EE3-201.PV"}]
# X_train = EE3_training_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV"}]
elif feature_num == "all":
    X_train = pd.concat([EE3_training_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV", "JEP.EU_WZIRA-EE3-201.PV"}], BD_training_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]], axis=1)
else:
    # X_train = BD_training_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]
    X_train = BD_training_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV"}]
# X_train = pd.concat([EE3_training_data[{"JEP.EU_DIRA-EE3-201.PV"}], BD_training_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]], axis=1)


# if model_num == 1:
#     y_train = error_training_data["error_training_act"] - error_training_data["error_training_cal"]
# elif model_num == 2:
#     y_train = error_training_data["error_training_act"]
y_train1 = error_training_data["error_training_act"] - error_training_data["error_training_cal"]
y_train2 = error_training_data["error_training_act"]
## RFモデル
random.seed(42)
np.random.seed(42)
# rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1, criterion='friedman_mse')
rf = MultiOutputRegressor(rf)
# rf2 = RandomForestRegressor(n_estimators=100, random_state=100, min_samples_split=2)
rf.fit(X_train, pd.concat([y_train1, y_train2], axis=1))
# rf2.fit(X_train, y_train2)
# print(rf)

#　推定
y_est = rf.predict(X_train)
## 可視化
draw_yield_fig(y_est[:, 0], y_train1, f'estimation_result', f'./output/RF/fig/est.png')

#　予測
## 検証データ
# X_ver = np.reshape(EE3_verification_data["JEP.EU_DIRA-EE3-201.PV"]._values, [-1,1])
if feature_num == "EE3":
    X_ver = EE3_verification_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV", "JEP.EU_WZIRA-EE3-201.PV"}]
# X_ver = EE3_verification_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV"}]
elif feature_num == "all":
    X_ver = pd.concat([EE3_verification_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV", "JEP.EU_WZIRA-EE3-201.PV"}], BD_verification_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]], axis=1)
else:
    # X_ver = BD_verification_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]
    X_ver = BD_verification_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV"}]
# X_ver = pd.concat([EE3_verification_data[{"JEP.EU_DIRA-EE3-201.PV"}], BD_verification_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]], axis=1)

# if model_num == 1:
#     y_act = error_verification_data["error_verification_act"] - error_verification_data["error_verification_cal"]
# elif model_num == 2:
#     y_act = error_verification_data["error_verification_act"]
y_act1 = error_verification_data["error_verification_act"] - error_verification_data["error_verification_cal"]
y_act2 = y_act = error_verification_data["error_verification_act"]
y_ver = rf.predict(X_ver)
## 可視化
draw_yield_fig(y_ver[:, 0], y_act1, f'verification_result', f'./output/RF/fig/ver.png')
## 真の温度を保存
BD_verification_data["JEP.EU_TIRA-BD-301.PV"].to_csv(f'./output/RF/file/T_real.csv')
BD_training_data["JEP.EU_TIRA-BD-301.PV"].to_csv(f'./output/RF/file/T_real_train.csv')
BD_data["JEP.EU_TIRA-BD-301.PV"].to_csv(f'./output/RF/file/T_real_all.csv')

# 適応物理モデル（物理モデル+誤差モデル）
# if model_num == 1:
#     yield_hybrid_ver = error_verification_data["error_verification_cal"] + y_ver
#     yield_hybrid_est = error_training_data["error_training_cal"] + y_est
#     ## 可視化
#     draw_yield_fig(yield_hybrid_ver, error_verification_data["error_verification_act"], f'hybrid_verification_result', f'./output/RF/fig/hybrid_verification.png')
#     draw_yield_fig(yield_hybrid_est, error_training_data["error_training_act"], f'hybrid_estimation_result', f'./output/RF/fig/hybrid_estimation.png')

#     # data_set = pd.concat([error_training_data["error_training_act"] - error_training_data["error_training_cal"],\
#     #                        EE3_training_data[{"JEP.EU_DIRA-EE3-201.PV", "JEP.EU_TIRA-EE3-201.PV", "JEP.EU_WZIRA-EE3-201.PV"}],\
#     #                          BD_training_data[{"JEP.EU_PIRCA-BD-101B.PV", "JEP.EU_TIRA-BD-301.PV", "JEP.EU_TIRCA-BD-101.PV"}]], axis=1)
#     # mat(data_set, f'./output/RF/fig/scatter.png')
#     draw_yield_fig2(error_verification_data["error_verification_cal"], yield_hybrid_ver, error_verification_data["error_verification_act"], f'verification_comparison', f'./output/RF/fig/verification_comparison.png')
# elif model_num ==2:
#     draw_yield_fig3(error_verification_data["error_verification_cal"], y_ver, error_verification_data["error_verification_act"], f'verification_comparison', f'./output/RF/fig/verification_comparison_statistical.png')

yield_hybrid_ver = error_verification_data["error_verification_cal"] + y_ver[:, 0]
yield_hybrid_est = error_training_data["error_training_cal"] + y_est[:, 0]
# draw_yield_fig(yield_hybrid_ver, error_verification_data["error_verification_act"], f'hybrid_verification_result', f'./output/RF/fig/hybrid_verification.png')
# draw_yield_fig(yield_hybrid_est, error_training_data["error_training_act"], f'hybrid_estimation_result', f'./output/RF/fig/hybrid_estimation.png')
draw_yield_fig4(error_verification_data["error_verification_cal"], y_ver[:, 1], yield_hybrid_ver, error_verification_data["error_verification_act"], f'verification_comparison', f'./output/RF/fig/verification_comparison_' + feature_num + '.png')
# X対yの散布図
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_ver["JEP.EU_TIRA-BD-301.PV"], y_ver[:, 0])
fig.savefig(f'./output/RF/fig/scatter_xy_' + feature_num + '.png')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_train["JEP.EU_TIRA-BD-301.PV"], y_train1)
fig.savefig(f'./output/RF/fig/scatter_xy_train_' + feature_num + '.png')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_train["JEP.EU_PIRCA-BD-101B.PV"], y_train1)
fig.savefig(f'./output/RF/fig/scatter_xy_train_P_' + feature_num + '.png')

# モデルオブジェクト保存
import pickle
with open(f'./output/RF/file/RF_model.pkl', 'wb') as file:
    pickle.dump(rf, file)