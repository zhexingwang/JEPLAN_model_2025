# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import os
from os import path

from pkg.subpkg.get_param import get_param_class
from pkg.subpkg.figure_save import figure_save_class
from pkg.subpkg.get_sol_loop import get_sol_loop_class
from pkg.subpkg.get_eval import get_obj_class
from pkg.subpkg.get_eval import get_vio_class
from pkg.function.function import Function


def make_setfile(dir_base, str_list, num_list):
    f = open(dir_base + '\\set_file.txt', 'w')
    for i in range(0,len(str_list)):
        f.write(str_list[i] + ' = ' + str(num_list[i]) + '\n')
    f.close()

    
def get_path(problem_type, N):
    work_path = path.dirname( path.abspath(__file__) )
    os.chdir(work_path)
    #work_path = os.path.abspath('..\\..\\') 
    dir_base = work_path + '\\output\\pro' + str(problem_type)
    my_makedirs(dir_base)
    my_makedirs(dir_base + "\\file")
    my_makedirs(dir_base + "\\fig")
    return dir_base

def get_parameter():
    work_path = path.dirname( path.abspath(__file__) )
    os.chdir(work_path)
    dir_base = work_path + '\\input\\'
    df = pd.read_csv(dir_base + 'parameter.csv', header=0)
    return df

def get_df_obj_vio(obj_vio_box, clm_list):
    m = obj_vio_box.shape[1]
    num_list = [str(n) for n in range(0, m)]
    for i in range(0, 2):
        num_list_ = [clm_list[i] + n for n in num_list]
        df_ = pd.DataFrame(obj_vio_box[:, :, i], columns=num_list_)
        if i == 0:
            df_obj_vio = df_.copy()
        else:
            df_obj_vio = pd.concat([df_obj_vio, df_], axis=1)
    return df_obj_vio


def get_result(obj_gbest_box, obj_optima):
    name_list = ["obj", "vio", "sum"]
    idx_list = ["ave", "std", "max", "min"]

    df_obj_gbest_final = pd.DataFrame(obj_gbest_box.T, columns=name_list)
    df_obj_gbest_final.loc[:, "obj-obj_optima"] = np.abs(df_obj_gbest_final.loc[:, "obj"].values - obj_optima)
    print(df_obj_gbest_final)

    name_list = df_obj_gbest_final.columns.tolist()
    df_result = pd.DataFrame(np.zeros((len(idx_list), len(name_list))), index=idx_list, columns=name_list)
    for i in name_list:
        df_result.at["ave", i] = df_obj_gbest_final.loc[:, i].mean()
        df_result.at["std", i] = df_obj_gbest_final.loc[:, i].std()
        df_result.at["max", i] = df_obj_gbest_final.loc[:, i].max()
        df_result.at["min", i] = df_obj_gbest_final.loc[:, i].min()


    print ("result=")
    print (df_result)
    return df_result, df_obj_gbest_final

def eval_method(func, str_, x_):
    print(str_ + ' = ')
    print(x_)
    f_ = func.object_function(x_)
    type_vio = 'max'
    (vio_, each_vio) = get_vio_class().get_vio(func, x_, type_vio)
    print('obj = ', f_)
    print('vio_sum = ', vio_)
    #print('each_vio=')
    #print(each_vio)

def test_eval(func, x_optima=np.array(0)):
    #eval_method(func, 'x_optima', x_optima)
    eval_method(func, 'x_all_zero', np.zeros(func.N))
    eval_method(func, 'x_all_one', np.ones(func.N))
    #pattern_x = np.delete(prob.pattern, np.where(prob.seedflag == 1))
    #eval_method(func, 'pattern', pattern_x)
    

    
def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

        
def main():
    # parameter setting
    (pro_para, alg_para, run_max) = get_param_class().get_param()
    #pro_para = [problem_type, d_exp, N]
    #alg_para = [alg_type, iter_max, m, "alg_name", **algorithm's other para.**]
    problem_type = pro_para[0]
    d_exp = pro_para[1]
    N = pro_para[2]

    DE_type = alg_para[0]
    eval_type = alg_para[1]
    iter_max = alg_para[2]
    m = alg_para[3]


    # instance
    func = Function(problem_type, N, DE_type, d_exp, eval_type)

    # g_fac : (Time*num_fac,)
    # g_demand : (Time*num_demand,)
    (x_optima, obj_optima) = func.get_opt()


    
    # test
    #test_eval(func)


    # get path
    dir_base = get_path(problem_type, func.N)
    str_clm = ['N','problem_type','DE_type','eval_type','iter_max','m','run_max','obj_optima']
    num_clm = [func.N,problem_type,DE_type,eval_type,iter_max,m,run_max,obj_optima]
    print(str_clm)
    print(num_clm)
    make_setfile(dir_base, str_clm, num_clm)

    # databox gene
    time_box = np.zeros((run_max+1, 3))
    # [gbest_obj, gbest_vio, gbest_eval]
    obj_gbest_box = np.zeros((iter_max, 3, run_max))
    x_gbest_final_box = np.zeros((func.N, run_max))
    # [F, CR]
    param_box = np.zeros((iter_max, 2, m, run_max))


    # run loop    
    # time start
    start_time = time.time()
    print ("calculation started.")
    print('-----------------------------------')

    for run in range(0, run_max): 
        print('run/run_max = %d / %d' % (run+1, run_max))
        print('loop start')
        # get solution
        (obj_subbox, x_gbest, obj_gbest_subbox, param_subbox) = get_sol_loop_class().main_loop(func, alg_para, run, start_time)

        # save solution
        obj_gbest_box[:, :, run] = obj_gbest_subbox.T
        x_gbest_final_box[:, run] = x_gbest
        param_box[:, :, :, run] = param_subbox[:, :, :].T
        
        now_time = time.time() 
        if run == 0:
            loop_cal_time = now_time - start_time
        else:
            loop_cal_time = now_time - cal_time - start_time
        cal_time = now_time - start_time 
        time_box[run, :] = np.array([loop_cal_time, loop_cal_time/60, loop_cal_time/3600])
        print('calculation time = %.3f sec = %.3f min' % (cal_time, cal_time/60))
        print ("f_gbest: ", np.round(obj_gbest_box[-1, :, run], 3))
        print ("x_gbest: ", np.round(x_gbest_final_box[:, run], 3))
        (vio_, each_vio) = get_vio_class().get_vio(func, x_gbest, 'sum')
        print ("each_vio: ", each_vio)
        print('loop end')
        print('-----------------------------------')

    print ("calculation finished")
    time_box[-1, :] = np.mean(time_box[:run_max, :], axis=0)
    time_box = np.round(time_box, decimals=4)
    print('mean time = %.3f sec = %.3f = min = %.3f = hour' % (time_box[-1, 0], time_box[-1, 1], time_box[-1, 2]))

    if "My Drive" in os.getcwd():
        file_base = dir_base + '/file/'
        fig_base = dir_base + '/fig/'
    else:
        file_base = dir_base + '\\file\\'
        fig_base = dir_base + '\\fig\\'


    # result save
    # df_result: [[ave, std, max, min], [obj, vio]]
    # df_obj_gbest_final: [run_max, [obj, vio]]
    run_idx = ['run'+str(i+1) for i in range(0, run_max)]
    df_time = pd.DataFrame(time_box, index=run_idx + ['average'], columns=['sec','min','hour'])
    df_time.to_csv(file_base + "time.csv")

    (df_result, df_obj_gbest_final) = get_result(obj_gbest_box[-1, :, :], obj_optima)
    df_result.to_csv(file_base + "result.csv")

    df_x_gbest_final = pd.DataFrame(x_gbest_final_box)
    df_x_gbest_final.to_csv(file_base + "x_gbest_final.csv")

    #obj_vio_box ((iter_max, m, 2, run_max))
    for run in range(0, run_max):
        df_feas_gbest = pd.DataFrame(obj_gbest_box[:, :, run], columns=['gbest_obj', 'gbest_vio', 'gbest_sum'])
        df_feas_gbest.to_csv(file_base + "gbest_run" + str(run) + ".csv")
        df_F = pd.DataFrame(param_box[:, 0, :, run])
        df_C = pd.DataFrame(param_box[:, 1, :, run])
        df_F.to_csv(file_base + "F_run" + str(run) + ".csv")
        df_C.to_csv(file_base + "C_run" + str(run) + ".csv")

    
    print ("file saving finished.")


    # figure save
    obj_max = np.amax(obj_gbest_box[:, 0, :])
    vio_max = np.amax(obj_gbest_box[:, 1, :])
    # [obj_min, obj_max, vio_min, vio_max]
    minmax = [0, np.abs(obj_max - obj_optima), 0, vio_max]
    #minmax = [0, obj_max]

    #figure_label1 = ["iteration: $k$", "$|$f(x)-f^{opt}$|$"] + minmax
    figure_label1 = ["iteration: $k$", "|$f(x)-f^{opt}$|", "$v(x)$", "|$f(x)-f^{opt}$|", "$v(x)$"] + minmax

    for run in range(0, run_max):
        # obj and vio gbest trend
        fig_file_name = fig_base + 'obj_vio_gbest_'+ str(run) + '.png'
        figure_save_class().double_trend(figure_label1, 
                                         range(0, iter_max), 
                                         obj_gbest_box[:, :, run],
                                         fig_file_name,
                                         scale1 = 'log',
                                         scale2 = 'log')
                

    print ("figure saving finished.")

    # time finish
    end_time = time.time()
    cal_time = end_time - start_time
    print('time = %f sec = %f min' % (cal_time, cal_time/60))

    
if __name__ == "__main__":
    main()
