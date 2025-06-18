#coding: utf-8
import numpy as np
import os
from os import path
import gc

from .import_data import import_data_class
from ..method import JADE
from ..method import SHADE
from ..method.SHADE.DE.DE_update import x_update_class
from ..method.SHADE.param_update import param_update_class
from ..method.JADE.DE.DE_update import x_update_class
from ..method.JADE.param_update import param_update_class
from ..method.modify_x import Modx
from .get_eval import Superior

class get_sol_loop_class:
    def main_loop(self, prob, alg_para, run, start_time):
        # get parameters
        # problem parameters
        N = prob.N
        problem_type = prob.problem_type

        # algorithm parameters
        DE_type = alg_para[0]
        iter_max = alg_para[2]
        m = alg_para[3]

        # 1. databox gene
        # obj: [obj, obj_eval, vio_sum, vio_eval]
        obj_box = np.zeros((m, iter_max))
        # obj_gbest_box: [obj, vio_sum, obj_eval]
        obj_gbest_box = np.zeros((3, iter_max))
        param_box = np.zeros((m, 2, iter_max))

        # 2. initial solution
        data = import_data_class().get_dataset(run, 2)
        #x = data[0:m, 0:N]*(prob.xmax-prob.xmin) + prob.xmin
        # x_ul: (N, 2), [min, max]
        x = data[:m, :N]*(prob.x_ul[:, 1]-prob.x_ul[:, 0]) + prob.x_ul[:, 0]

        del data
        gc.collect()


        work_path = path.dirname( path.abspath(__file__) )
        os.chdir(work_path)


        # get obj: (m, 4)
        (obj, each_vio) = Superior().eval_scaler(prob, x)

        # vio = sigma_{j=1, L} orm_j
        # vio_eval = sigma_{j=1, L} orm_j
        # vio_eval = sigma_{j=1, L} (orm_max - orm_j / orm_max - orm_min)
        # orm_max = max_{i=1,...,m} orm_j
        # orm_min = min_{i=1,...,m} orm_j
        # orm_j : (m, L) [0, g_j(x)]

        # mod_type: round, reflect, torus
        mod_type = 'round'
        mod = Modx(prob.x_ul, type_mod=mod_type)

        idx_gbest = Superior().get_min_idx_array(obj, prob.eval_type)
        x_gbest = mod.modified_x(x[idx_gbest, :].copy())
        (obj_gbest, each_vio) = Superior().eval_scaler(prob, x_gbest)
        print('inital obj_gbest = ', obj_gbest[:3])

        # 3. iteration loop                
        iter = 0
        s_F = []
        s_C = []
        if DE_type == 1:
            DE_x_update_instance = JADE.DE.DE_update.x_update_class(m, N)
            DE_param_update_instance = JADE.param_update.param_update_class(m)
            (DE_x_update_instance.F, DE_x_update_instance.C) = DE_param_update_instance.update_param(s_F, s_C)
        elif DE_type == 2:
            DE_x_update_instance = SHADE.DE.DE_update.x_update_class(m, N)
            DE_param_update_instance = SHADE.param_update.param_update_class(m)
            (DE_x_update_instance.F, DE_x_update_instance.C) = DE_param_update_instance.update_param(s_F, s_C, iter)

        while (iter <= iter_max - 1):
            # 4. solution update
            (x, obj) = DE_x_update_instance.DE_update(prob, x, obj, mod)
            s_F = DE_x_update_instance.F[DE_x_update_instance.update_idx_]
            s_C = DE_x_update_instance.C[DE_x_update_instance.update_idx_]
            if DE_type == 1:
                (DE_x_update_instance.F, DE_x_update_instance.C) = DE_param_update_instance.update_param(s_F, s_C)
            elif DE_type == 2:
                (DE_x_update_instance.F, DE_x_update_instance.C) = DE_param_update_instance.update_param(s_F, s_C, iter)
            idx_gbest = Superior().get_min_idx_array(obj, prob.eval_type)
            x_gbest = mod.modified_x(x[idx_gbest, :].copy())
            (obj_gbest, each_vio) = Superior().eval_scaler(prob, x_gbest)
            
            # other data update
            if prob.eval_type <= 3:
                clm = Superior().get_clm(prob.eval_type)
                obj_box[:, iter] = obj[:, clm].copy()
            else:
                obj_box[:, iter] = obj[:, 1].copy()
            obj_gbest_box[:, iter] = obj_gbest[:3].copy()
            param_box[:, :, iter] = np.array([DE_x_update_instance.F, DE_x_update_instance.C]).T

            iter = iter + 1

        return obj_box, x_gbest, obj_gbest_box, param_box