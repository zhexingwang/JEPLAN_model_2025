import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from model import antoine, nrtl
def const(T_input, b_input, d_input, c_input):
    ec = []
    fc = []
    gc = []
    yc = []
    hc = []
    g1c = []
    g2c = []
    g3c = []
    g4c = []
    g5c = []
    g6c = []
    for i in range(len(T_input)):
        ec.append(NonlinearConstraint(lambda x: antoine(T_input[i], x[0], x[1], x[2]), 0, 20))
        fc.append(NonlinearConstraint(lambda x: antoine(T_input[i], x[3], x[4], x[5]), 0, 20))
        gc.append(NonlinearConstraint(lambda x: nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0], 0, 1))
        yc.append(NonlinearConstraint(lambda x: ( (b_input[i]/100) * d_input[i] / (d_input[i] - antoine(T_input[i], x[3], x[4], x[5])*nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1]) ) + ( ((c_input[i])/100) * d_input[i] / (d_input[i] - antoine(T_input[i], x[0], x[1], x[2])*nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0]) ),0, 1))
        hc.append(NonlinearConstraint(lambda x: nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1], 0, 1))
        # g1の制約
        g1c.append(NonlinearConstraint(lambda x: (-antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] + d_input[i]) / (antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] - antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1]) ,0, 1))
        # g2の制約
        g2c.append(NonlinearConstraint(lambda x: (antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] - d_input[i]) / (antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] - antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1]) , 0, 1))
        # g3の制約
        g3c.append(NonlinearConstraint(lambda x: (antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] * (-antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] +d_input[i])) / ((antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] - antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1]) * d_input[i]) , 0, 1))
        # g4の制約
        g4c.append(NonlinearConstraint(lambda x: (-antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] * (-antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] + d_input[i])) / ((antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] - antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1]) * d_input[i]) , 0, 1))
        # g5の制約
        g5c.append(NonlinearConstraint(lambda x: (-a_input[i] * (-(b_input[i]/100) * antoine(T_input[i], x[0], x[1], x[2]) * antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] - (c_input[i]/100) * antoine(T_input[i], x[0], x[1], x[2]) * antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] + (b_input[i] / 100) * d_input[i] * antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] + (c_input[i]/100) * d_input[i] * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1])) / ((-antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] + d_input[i]) * (-antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] + d_input[i])), 0, np.inf))
        # g6の制約
        g6c.append(NonlinearConstraint(lambda x: ((-(b_input[i]/100) * antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] - (c_input[i]/100) * antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] + (b_input[i] /100) * d_input[i] + (c_input[i]/100) * d_input[i]) * a_input[i] * d_input[i]) / ((antoine(T_input[i], x[0], x[1], x[2]) * antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] - d_input[i] * antoine(T_input[i], x[0], x[1], x[2]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[0] -d_input[i] * antoine(T_input[i], x[3], x[4], x[5]) * nrtl(T_input[i]+273.15, x[6], x[7], x[8], x[9], np.asarray([b_input[i]/100,c_input[i]/100]))[1] + d_input[i]**2)) , 0, np.inf))
    return ec+fc+gc+hc+yc+g1c+g2c+g3c+g4c+g5c+g6c