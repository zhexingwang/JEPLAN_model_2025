import numpy as np
import model

def load_param(fitting_data, M1, M2, conversion=False):
    z1_mol = fitting_data['feed_com'].to_numpy() /100
    z1 = model.z1_input(fitting_data, M1, M2, conversion=conversion)
    p_total = fitting_data['JEP.EU_PIRCA-BD-101B.PV'].to_numpy()
    T = fitting_data['JEP.EU_TIRA-BD-301.PV'].to_numpy()
    F_in = fitting_data['JEP.EU_WZIRA-EE3-201.PV'].to_numpy()
    yield_act = fitting_data['yield'].to_numpy()    
    return z1_mol, z1, p_total, T, F_in, yield_act

def initialize_s():
    '''
    Antoine eq. (1st component) s1 = A, s2 = B, s3 = C
    Antoine eq. (2nd component) s4 = A, s5 = B, s6 = C
    NRTL eq.    (common)        s7 = alpha
    NRTL eq.    (common)        s8 = Δg12, s9 = Δg21
    '''
    s1, s2, s3 = 6, 1200, 230
    s4, s5, s6 = 6, 1200, 230
    s7 = 0.3
    s8, s9 = -0.08, 355
    return [s1, s2, s3, s4, s5, s6, s7, s8, s9]

def def_bounds():
    bound_s1, bound_s2, bound_s3 = (5, 6), (1000, 2000), (225, 235) # A, B, C
    bound_s4, bound_s5, bound_s6 = (5, 6), (1000, 2000), (225, 235) # A, B, C
    # bound_s1, bound_s2, bound_s3 = (5, 6), (1000, 2000), (200, 250) # A, B, C
    # bound_s4, bound_s5, bound_s6 = (5, 6), (1000, 2000), (200, 250) # A, B, C
    bound_s7 = (0.2, 0.47) # alpha
    # bound_s7 = (-1, 1) # alpha
    bound_s8, bound_s10 = (-5000, 5000), (-5000, 5000) # Δg12, Δg21
    bounds = [bound_s1, bound_s2, bound_s3, bound_s4, bound_s5, bound_s6, bound_s7, bound_s8, bound_s10]
    return bounds

#評価関数
class obj:
    def __init__(self, z1_mol, z1, p_total, M1, M2, F_in, T, yield_act, get_each_yield_cal, abs_diff):
        self.z1_mol = z1_mol
        self.z1 = z1
        self.p_total = p_total
        self.M1 = M1
        self.M2 = M2
        self.F_in = F_in 
        self.T = T
        self.yield_act = yield_act
        # self.yield_cal = yield_cal
        self.abs_diff = abs_diff
        self.get_each_yield_cal = get_each_yield_cal
        # self.get_each_const = get_each_const
        self.best_x = None
        self.minf = np.inf

    def __call__(self, s):
        f = self.E(s)        
        if f < self.minf:
            self.minf = f
            self.best_s = s
        return f

    def E(self, S): # S:最適化するパラメータベクトル
        cal_array = self.get_each_yield_cal(self.z1_mol, self.z1, self.p_total, self.M1, self.M2, self.F_in, S, self.T)
        return np.sqrt(self.abs_diff(self.yield_act, cal_array).mean())
