import numpy as np
import pandas as pd
import math
from scipy.constants import calorie, R, atm
from scipy.optimize import differential_evolution, NonlinearConstraint
import opt_problem_JEPLAN as opt_problem


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


    def p1_lb(self, x):
        return -1 * self.p1(x) + self.lowerupper[0][0]
    def p2_lb(self, x):
        return -1 * self.p2(x) + self.lowerupper[1][0]
    def gamma1_lb(self, x):
        return -1 * self.gamma1(x) + self.lowerupper[2][0]
    def gamma2_lb(self, x):
        return -1 * self.gamma2(x) + self.lowerupper[3][0]
    def x1_lb(self, x):
        return -1 * self.x1(x) + self.lowerupper[4][0]
    def y1_lb(self, x):
        return -1 * self.y1(x) + self.lowerupper[5][0]
    def w1_liquid_lb(self, x):
        return -1 * self.w1_liquid(x) + self.lowerupper[6][0]
    def w1_gas_lb(self, x):
        return -1 * self.w1_gas(x) + self.lowerupper[7][0]
    def F_liquid_lb(self, x):
        return -1 * self.F_liquid(x) + self.lowerupper[8][0]
    def F_gas_lb(self, x):
        return -1 * self.F_gas(x) + self.lowerupper[9][0]
    def yield__lb(self, x):
        return -1 * self.yield_(x) + self.lowerupper[10][0]
    def FF_lb(self, x):
        return -1 * self.FF(x) + self.lowerupper[11][0]
    
    def p1_ub(self, x):
        return self.p1(x) - self.lowerupper[0][1]
    def p2_ub(self, x):
        return self.p2(x) - self.lowerupper[1][1]
    def gamma1_ub(self, x):
        return self.gamma1(x) - self.lowerupper[2][1]
    def gamma2_ub(self, x):
        return self.gamma2(x) - self.lowerupper[3][1]
    def x1_ub(self, x):
        return self.x1(x) - self.lowerupper[4][1]
    def y1_ub(self, x):
        return self.y1(x) - self.lowerupper[5][1]
    def w1_liquid_ub(self, x):
        return self.w1_liquid(x) - self.lowerupper[6][1]
    def w1_gas_ub(self, x):
        return self.w1_gas(x) - self.lowerupper[7][1]
    def F_liquid_ub(self, x):
        return self.F_liquid(x) - self.lowerupper[8][1]
    def F_gas_ub(self, x):
        return self.F_gas(x) - self.lowerupper[9][1]
    def yield__ub(self, x):
        return self.yield_(x) - self.lowerupper[10][1]
    def FF_ub(self, x):
        return self.FF(x) - self.lowerupper[11][1]

    def const_cal_jDE_(self, i, z1_mol, z1, p_total, T, F_in):
        def update_call(method):
            self.z1_mol = z1_mol[i]
            self.T = T[i]
            self.p_total = p_total[i]
            self.z1 = z1[i]
            self.F_in = F_in[i]
            method()
        # return [self.p1_lb, self.p1_ub,\
        #         self.p2_lb, self.p2_ub,\
        #         self.gamma1_lb, self.gamma1_ub,\
        #         self.gamma2_lb, self.gamma2_ub,\
        #         self.x1_lb, self.x1_ub,\
        #         self.y1_lb, self.y1_ub,\
        #         self.w1_liquid_lb, self.w1_liquid_ub,\
        #         self.w1_gas_lb, self.w1_gas_ub,\
        #         self.F_liquid_lb, self.F_liquid_ub,\
        #         self.F_gas_lb, self.F_gas_ub,\
        #         self.yield__lb, self.yield__ub,\
        #         self.FF_lb, self.FF_ub]
        return [lambda: update_call(self.x1_lb), lambda: update_call(self.x1_ub),\
                lambda: update_call(self.y1_lb), lambda: update_call(self.y1_ub)]
        # return [self.p1_lb, self.p1_ub,\
        #         self.x1_lb, self.x1_ub,\
        #         self.y1_lb, self.y1_ub]

    def get_each_const_jDE(self, fitting_data, M1, M2): #ステップ毎に制約条件を作成してappend
        z1_mol, z1, p_total, T, F_in, yield_act = opt_problem.load_param(fitting_data, M1, M2, conversion=self.conversion)
        constr = []
        #ローカル変数iを作る

        for i in range(len(z1_mol)):
            "修正中"
            self.z1_mol = z1_mol[i]
            self.T = T[i]
            self.p_total = p_total[i]
            self.z1 = z1[i]
            self.F_in = F_in[i]
            # self.yield_act = yield_act[i]
            constr.append(self.const_cal_jDE_(i, z1_mol, z1, p_total, T, F_in))
            "修正中"
        return np.concatenate(np.array(constr))

def get_each_yield_cal(z1_mol, z1, p_total, M1, M2, F_in, S, T):
    return np.array([yield_cal(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8], T[i]) for i in range(len(z1_mol))])
    # return np.array([yield_cal(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], S, T[i]) for i in range(len(z1_mol))])
# def get_each_const(z1_mol, z1, p_total, M1, M2, F_in, T):
#     return np.array([cst.const_cal_(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], T[i]) for i in range(len(z1_mol))])

def get_each_yield_cal_all(z1_mol, z1, p_total, M1, M2, F_in, S, T):
    ret = np.array([yield_cal(z1_mol[i], z1[i], p_total[i], M1, M2, F_in[i], S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8], T[i], return_vars=True) for i in range(len(z1_mol))])
    return ret

def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

def abs_diff(act, est):
    try:
        return np.power(act - est, 2)
    except:
        est = np.array([e._value for e in est])
        return np.power(act - est, 2)


class my_constrained_udp:
    def __init__(self, obj, const):
        self.obj = obj
        self.const = const
        self.ni = len(const)
    def fitness(self, x):
        consts = []
        for i in range(len(self.const)):
            consts.append(self.const[i](x))
        return [self.obj.E(x)] + consts
    def get_bounds(self):
        return opt_problem.def_bounds_jDE()
    def get_nic(self):
        return self.ni
    def get_nec(self):
        return 0
