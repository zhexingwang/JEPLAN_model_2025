import numpy as np
import pandas as pd
import math
from scipy.constants import calorie, R, atm
from scipy.optimize import differential_evolution, NonlinearConstraint
import opt_problem


def convert_units(x, M1, M2):
    w = (M1 * x) / (M1 * x + M2 * (1 - x))
    return w

def z1_input(fitting_data, M1, M2, conversion=True):
    if conversion:
        z1_mol = fitting_data['feed_com'].to_numpy() /100
        z1 = convert_units(z1_mol, M1, M2)
    else:
        z1 = fitting_data['feed_com'].to_numpy() /100
    return z1

# def lee_kesler(T):
#     sigma_Ntcbk = -0.3899
#     sigma_Npcbk = -0.2754
#     N_atoms = 32
#     Tb = 702.12 # T(K)
#     Tc = Tb / (0.5851 - 0.9286*(sigma_Ntcbk) - (sigma_Ntcbk)**2)
#     Pc = ((0.1285 -0.0059*N_atoms - sigma_Npcbk)**-2) / 1.01325
#     Tr = (T + 273.15) / Tc
#     sita = Tb / Tc
#     w = (-np.log(Pc) - f0_(sita)) / f1_(sita)
#     f0 = 5.92714 - (6.09648/Tr) - 1.28862*np.log(Tr) + 0.169347*((Tr)**6)
#     f1 = 15.2518 - (15.6875/Tr) - 13.4721*np.log(Tr) + 0.43577*((Tr)**6)
#     pvr = np.exp(f0 + w * f1)
#     # pvr = np.exp(f0(Tr) + w(Tb, Tc, Pc)*f1(Tr))
#     svp = pvr * Pc
#     return svp * 101.325 * 10**3

def antoine(T, A, B, C):
    p = 10**(A - (B / (T + C)))
    return p

def nrtl_binary(z1, T, R, s7, s8, s9):
    tau12 = s8 / (R * (T + 273.15))
    tau21 = s9 / (R * (T + 273.15))
    G12 = math.exp(-s7 * tau12)
    G21 = math.exp(-s7 * tau21)
    gamma1 = math.exp((1 - z1)**2 * (tau21 * ((G21 / (z1 + (1 - z1) * G21))**2) 
                                        + (tau12 * G12) / ((1 - z1) + z1 * G12)**2))
    gamma2 = math.exp(z1**2 * (tau12 * ((G12 / ((1 - z1) + z1 * G12))**2) 
                                 + (tau21 * G21) / (z1 + (1 - z1) * G21)**2))
    return [gamma1, gamma2]

# 全段階を順に計算：式(23)
def yield_cal(z1_mol, z1, p_total, M1, M2, F_in, s1, s2, s3, s4, s5, s6, s7, s8, s9, T, return_vars=False):
    p1 = antoine(T, s1, s2, s3)
    p2 = antoine(T, s4, s5, s6)
    gamma1 = nrtl_binary(z1_mol, T, R, s7, s8, s9)[0]
    gamma2 = nrtl_binary(z1_mol, T, R, s7, s8, s9)[1]
    x1 = (p_total - p2 * gamma2) / (p1 * gamma1 - p2 * gamma2)
    y1 = (p1 * gamma1 / p_total) * x1
    w1_liquid = convert_units(x1, M1, M2)
    w1_gas = convert_units(y1, M1, M2)
    F_liquid = F_in * ((z1 - w1_gas) / (w1_liquid - w1_gas))
    F_gas = F_in * ((w1_liquid - z1) / (w1_liquid - w1_gas))
    yield_cal = F_gas / (F_liquid + F_gas)
    if return_vars:
        return yield_cal, p1, p2, gamma1, gamma2, x1, y1, w1_liquid, w1_gas, F_liquid, F_gas
    return yield_cal

'''
　　中間変数の計算式
    p1 = antoine(T, s1, s2, s3)
    p2 = antoine(T, s4, s5, s6)
    gamma1 = nrtl_binary(z1_mol, T, R, s7, s8, s9)[0]
    gamma2 = nrtl_binary(z1_mol, T, R, s7, s8, s9)[1]
    x1 = (p_total - p2 * gamma2) / (p1 * gamma1 - p2 * gamma2)
    y1 = (p1 * gamma1 / p_total) * x1
    w1_liquid = convert_units(x1, M1, M2)
    w1_gas = convert_units(y1, M1, M2)
    F_liquid = F_in * ((z1 - w1_gas) / (w1_liquid - w1_gas))
    F_gas = F_in * ((w1_liquid - z1) / (w1_liquid - w1_gas))
    yield_cal = F_gas / (F_liquid + F_gas)
    w1_liquid - w1_gas > 0
'''

# 制約条件クラス
class cst:
    def __init__(self, fitting_data, M1, M2, conversion=False):
        self.conversion = conversion
        z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(fitting_data, M1, M2, conversion=self.conversion)
        self.z1_mol = z1_mol
        self.T = T
        self.p_total = p_total
        self.z1 = z1
        self.F_in = F_in
        self.yield_act = yield_act
        self.M1 = M1
        self.M2 = M2
        # a = self.const_cal_()
        self.g_num = 12
        self.lowerupper = [[0, 0] for i in range(self.g_num)]
        # lower
        self.lowerupper[0][0] = 0
        self.lowerupper[1][0] = 0
        self.lowerupper[2][0] = 0
        self.lowerupper[3][0] = 0
        self.lowerupper[4][0] = 0
        self.lowerupper[0][0] = 0.001
        self.lowerupper[1][0] = 0.001
        self.lowerupper[2][0] = 0.001
        self.lowerupper[3][0] = 0.001
        self.lowerupper[4][0] = 0.001
        self.lowerupper[5][0] = 0.001
        self.lowerupper[6][0] = 0.001
        self.lowerupper[7][0] = 0.001
        self.lowerupper[8][0] = 0.001
        self.lowerupper[9][0] = 0.001
        self.lowerupper[10][0] = 0
        self.lowerupper[11][0] = 0
        # upper
        self.lowerupper[0][1] = 20
        self.lowerupper[1][1] = 20
        self.lowerupper[2][1] = 1
        self.lowerupper[3][1] = 1
        self.lowerupper[4][1] = 1
        self.lowerupper[5][1] = 1
        self.lowerupper[6][1] = 1
        self.lowerupper[7][1] = 1
        self.lowerupper[8][1] = 152
        self.lowerupper[9][1] = 152
        self.lowerupper[10][1] = 1
        self.lowerupper[11][1] = 152

    def p1(self, x):
        return antoine(self.T, x[0], x[1], x[2])
    def p2(self, x):
        return antoine(self.T, x[3], x[4], x[5])
    def gamma1(self, x):
        return nrtl_binary(self.z1_mol, self.T, R, x[6], x[7], x[8])[0] # nrtl_binary中には既に計算されているため、+273.15を削除
    def gamma2(self, x):
        return nrtl_binary(self.z1_mol, self.T, R, x[6], x[7], x[8])[1] # nrtl_binary中には既に計算されているため、+273.15を削除
    def x1(self, x):
        return (self.p_total - self.p2(x) * self.gamma2(x)) / (self.p1(x) * self.gamma1(x) - self.p2(x) * self.gamma2(x))
    def y1(self, x):
        'y1 = (p1 * gamma1 / p_total) * x1'
        return (self.p1(x) * self.gamma1(x) / self.p_total) * self.x1(x)
    def w1_liquid(self, x):
        'w1_liquid = convert_units(x1, M1, M2)'
        return convert_units(self.x1(x), self.M1, self.M2)
    def w1_gas(self, x):
        'w1_gas = convert_units(y1, M1, M2)'
        return convert_units(self.y1(x), self.M1, self.M2)
    def F_liquid(self, x):
        'F_liquid = F_in * ((z1 - w1_gas) / (w1_liquid - w1_gas))'
        return self.F_in * ((self.z1 - self.w1_gas(x)) / (self.w1_liquid(x) - self.w1_gas(x)))
    def F_gas(self, x):
        'F_gas = F_in * ((w1_liquid - z1) / (w1_liquid - w1_gas))'
        return self.F_in * ((self.w1_liquid(x) - self.z1) / (self.w1_liquid(x) - self.w1_gas(x)))
    def yield_(self, x):
        'yield_cal = F_gas / (F_liquid + F_gas)'
        return self.F_gas(x) / (self.F_liquid(x) + self.F_gas(x))
    def FF(self, x):
        return self.F_liquid(x) + self.F_gas(x)

    def const_cal_(self):
        return [NonlinearConstraint(self.p1, self.lowerupper[0][0], self.lowerupper[0][1]),\
                NonlinearConstraint(self.p2, self.lowerupper[1][0], self.lowerupper[1][1]),\
                NonlinearConstraint(self.gamma1, self.lowerupper[2][0], self.lowerupper[2][1]),\
                NonlinearConstraint(self.gamma2, self.lowerupper[3][0], self.lowerupper[3][1]),\
                NonlinearConstraint(self.x1, self.lowerupper[4][0], self.lowerupper[4][1]),\
                NonlinearConstraint(self.y1, self.lowerupper[5][0], self.lowerupper[5][1]),\
                NonlinearConstraint(self.w1_liquid, self.lowerupper[6][0], self.lowerupper[6][1]),\
                NonlinearConstraint(self.w1_gas, self.lowerupper[7][0], self.lowerupper[7][1]),\
                NonlinearConstraint(self.F_liquid, self.lowerupper[8][0], self.lowerupper[8][1]),\
                NonlinearConstraint(self.F_gas, self.lowerupper[9][0], self.lowerupper[9][1]),\
                NonlinearConstraint(self.yield_, self.lowerupper[10][0], self.lowerupper[10][1]),\
                NonlinearConstraint(self.FF, self.lowerupper[11][0], self.lowerupper[11][1])]
    
    def get_each_const(self, fitting_data, M1, M2): #ステップ毎に制約条件を作成してappend
        z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(fitting_data, M1, M2, conversion=self.conversion)
        constr = []
        for i in range(len(z1_mol)):
            self.z1_mol = z1_mol[i]
            self.T = T[i]
            self.p_total = p_total[i]
            self.z1 = z1[i]
            self.F_in = F_in[i]
            # self.yield_act = yield_act[i]
            constr.append(self.const_cal_())
        return np.concatenate(np.array(constr))


def get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, S, T):
    return np.array([yield_cal(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8], T[i]) for i in range(len(z1_mol))])
    # return np.array([yield_cal(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], S, T[i]) for i in range(len(z1_mol))])
# def get_each_const(z1_mol, z1, p_total, M1, M2, F_in, T):
#     return np.array([cst.const_cal_(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], T[i]) for i in range(len(z1_mol))])

def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

def abs_diff(act, est):
    try:
        return np.power(act - est, 2)
    except:
        est = np.array([e._value for e in est])
        return np.power(act - est, 2)

# def nrtl_multi(T, alpha1, alpha2, g1, g2, x):
#     '''
#     NRTLによる活量係数の計算

#     Parameters
#     ----------
#     alpha : ndarray(n,n)
#         Array of NRTL nonrandomness parameters. n = the number of 
#         components in the system.
#     tau : ndarray(n,n)
#         Array of NRTL tau parameters. tau[i,i] should be set to 0.
#     t : float
#         Temperature (K)
#     x : ndarray(n,)
#         Mole fraction of each component

#     Returns
#     -------
#     gamma : ndarray(n,)
#         Activity coefficient of each component    
#     '''
#     try:
#         alpha = np.array([[0, alpha1],
#                             [alpha2, 0]])
#         g = np.asarray([[0, g1],
#                         [g2, 0]])
#         tau = g / (R * T)
#         G = np.exp(-(alpha * tau))
#     except TypeError:
#         # alpha1, alpha2, g1, g2, T = alpha1._value, alpha2._value, g1._value, g2._value, T._value
#         alpha = np.array([[0, alpha1],
#                             [alpha2, 0]])
#         g = np.asarray([[0, g1],
#                         [g2, 0]])
#         tau = g / (R * T)
#         G = np.exp(-(alpha * tau))
#     ncomp = x.shape[0]
#     gamma = np.zeros_like(x)
#     summ = 0
        
#     # ori
#     for i in range(ncomp):
#         summ = 0
#         for j in range(ncomp):
#             summ += x[j] * G[i,j] / np.sum(G[:,j] * x) * (tau[i,j] - (np.sum(x*tau[:,j] * G[:,j]) / np.sum(G[:,j] * x)))
#         gamma[i] = np.sum(tau[:,i] * G[:,i] * x) / np.sum(G[:,i] * x) + summ
#         # print(ncomp, i, gamma)
#     return np.exp(gamma)

# # multi分解、手計算
# def gamma(x, G, tau):
#     gamma = np.zeros_like(x)
#     for i in range(x.shape[0]):
#         summ = 0
#         for j in range(x.shape[0]):
#             summ += x[j] * G[i,j] / np.sum(G[:,j] * x) * (tau[i,j] - (np.sum(x*tau[:,j] * G[:,j]) / np.sum(G[:,j] * x)))
#         gamma[i] = np.sum(tau[:,i] * G[:,i] * x) / np.sum(G[:,i] * x) + summ
#     return np.exp(gamma)

# def nrtl2(T, alpha1, alpha2, g1, g2, x):
#     try:
#         alpha = np.array([[0, alpha1],
#                             [alpha2, 0]])
#         g = np.asarray([[0, g1],
#                         [g2, 0]])
#         tau = g / (R * T)
#         G = np.exp(-(alpha * tau))
#     except TypeError:
#         alpha1, alpha2, g1, g2, T = alpha1._value, alpha2._value, g1._value, g2._value, T._value
#         alpha = np.array([[0, alpha1],
#                             [alpha2, 0]])
#         g = np.asarray([[0, g1],
#                         [g2, 0]])
#         tau = g / (R * T)
#         G = np.exp(-(alpha * tau))
#     return gamma(x, G, tau)
