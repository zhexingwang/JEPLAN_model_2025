#coding: utf-8
import numpy as np
import math

import opt_problem_JEPLAN   # for parameter fitting
import model_JEPLAN         # for parameter fitting
# import opt_problem_JEPLAN2 as opt_problem_JEPLAN      # for optimizing T
# import model_JEPLAN2 as model_JEPLAN                  # for optimizing T
import load_files_JEPLAN

class Function:
    def __init__(self, problem_type, N, DE_type, d_exp, eval_type):
        self.problem_type = problem_type
        self.N = N
        self.PSO_type = DE_type
        self.eval_type = eval_type
        self.d = pow(10, d_exp)
        # self.x_ul = np.ones((self.N, 2))
        # self.x_ul[:, 0] = -5 * self.x_ul[:, 0]
        # self.x_ul[:, 1] = 5 * self.x_ul[:, 1]
        # self.x_ul[:, 0] = 0 * self.x_ul[:, 0]
        # self.x_ul[:, 1] = 5 * self.x_ul[:, 1]
        M1 = 254.24
        M2 = 400 # 200~500 g/mol
        unit_conversion = True
        print('import files')
        df, fitting_df, fitting_data = load_files_JEPLAN.load_files()
        z1_mol, z1, p_total, T, F_in, yield_act = opt_problem_JEPLAN.load_param(fitting_data, M1, M2, conversion=unit_conversion)
        self.x_ul = opt_problem_JEPLAN.def_bounds_SHADE()
        self.obj = opt_problem_JEPLAN.obj(z1_mol, z1, p_total, M1, M2, F_in, T, yield_act, model_JEPLAN.get_each_yield_cal, model_JEPLAN.abs_diff)
        self.Const = model_JEPLAN.cst(fitting_data, M1, M2, conversion=False)
        self.const = self.Const.get_each_const_jDE(fitting_data, M1, M2)
        self.T = T

    def get_opt(self):
        if self.problem_type <= 3:
            x_opt = (1-math.sqrt(self.d))*np.ones(self.N)
        elif self.problem_type == 4:
            x_opt = (0.25-math.sqrt(self.d))*np.ones(self.N)
        elif self.problem_type == 5:
            x_opt = np.ones(self.N)
            # x_opt[0] = 1.5
            # x_opt[1] = 3
            # x_opt[0] = -0.25
            # x_opt[1] = -0.9375
            x_opt[0] = 5.000265367
            x_opt[1] = 1485.575616
            x_opt[2] = 229.3702312
            x_opt[3] = 5.115318701
            x_opt[4] = 1969.071531
            x_opt[5] = 231.8896059
            x_opt[6] = 0.283637084
            x_opt[7] = -852.8665611
            x_opt[8] = 913.2134476
        elif self.problem_type == 6:
            x_opt = np.ones(self.N)
        else:
            x_opt = np.zeros(self.N)
            print('problem number is not preset.')
        # f_opt = pow(x_opt[0], 2)
        # f_opt = 180*x_opt[0]+160*x_opt[1]
        # f_opt = np.exp(x_opt[0]+2*x_opt[1])
        f_opt = self.obj.E(x_opt)
        return x_opt, f_opt

    def object_function(self, x__):
        if self.problem_type <= 4:
            return np.power(x__, 2).mean()
        elif self.problem_type == 5:
            # return 180*x__[0]+160*x__[1]
            # return np.exp(x__[0]+2*x__[1])
            return self.obj.E(x__)
        elif self.problem_type == 6:
            return self.obj.E(x__)

    def constraint_function(self, x__):
        def _g1(x__):
            return np.power(x__-1, 2).mean() - self.d
        def _g4(x__):
            return np.cos(2*np.pi*(x__-0.25)).mean() - math.cos(2*math.pi*math.sqrt(self.d))

        if self.problem_type == 1:
            return _g1(x__)
        elif self.problem_type == 2:
            return np.exp(10*_g1(x__))-1
        elif self.problem_type == 3:
            a = _g1(x__)
            return np.sign(a) * pow(abs(a), 1/4)
        elif self.problem_type == 4:
            return -1*_g4(x__)
        elif self.problem_type == 5:
            # ic1 = -6*x__[0]-x__[1]+12     # 6*x[0]+x[1]>=12
            # ic2 = -4*x__[0]-6*x__[1]+24   # 4*x[0]+6*x[1]>=24
            # ic1 = np.power(x__[0],2)/9+np.power(x__[1],2)/4-1   # x[0]^2/9+x[1]^2/4<=1
            # ic2 = np.power(x__[0],2)-x__[1]-1                   # x[1]>=x[0]^2-1
            # return np.array([ic1,ic2])
            consts = []
            for i in range(len(self.const)):
                consts.append(self.const[i](x__))
            return np.array(consts)
        elif self.problem_type == 6:
            consts = []
            for i in range(len(self.const)):
                consts.append(self.const[i](x__))
            return np.array(consts)
        else:
            return 0
