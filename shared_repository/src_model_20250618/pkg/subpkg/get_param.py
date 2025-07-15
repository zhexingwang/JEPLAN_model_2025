#coding: utf-8 
import re
import os
from os import path

class get_param_class:
    def get_param(self):
        work_path = path.dirname( path.abspath(__file__) )
        os.chdir(work_path)
        param_file = path.abspath('..\\..\\') + "\\alg_param.ini"
        with open(param_file, 'r') as f:
            for line in f:
                line = re.sub(r'\s|\n', '', line)
                if len(line) == 0 or line[0] == '#':
                    continue
                line = line.split("=")
                
                if line[0] == 'problem_type':
                    problem_type = int(line[1])
                elif line[0] == 'd_exp':
                    d_exp = int(line[1])
                elif line[0] == 'N':
                    N = int(line[1])
                elif line[0] == 'DE_type':
                    DE_type = int(line[1])
                elif line[0] == 'eval_type':
                    eval_type = int(line[1])
                elif line[0] == 'm':
                    # population size
                    m = int(line[1])
                elif line[0] == 'iter_max':
                    # iteration counter max
                    iter_max = int(line[1])
                elif line[0] == 'run_max':
                    # loop max
                    run_max = int(line[1])
                else:
                    raise 'invalid keyword: ' + line[0]
         
        # redefine
        alg_para = []
        pro_para = []


        # problem parameters
        #[problem_type, d_exp, N]
        pro_para.append(problem_type)
        pro_para.append(d_exp)
        pro_para.append(N)
        
        #[alg_type, iter_max, m, **algorithm's other para.**]
        alg_para.append(DE_type)
        alg_para.append(eval_type)
        alg_para.append(iter_max)
        alg_para.append(m)

        return pro_para, alg_para, run_max
