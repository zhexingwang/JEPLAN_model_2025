#coding: utf-8
import numpy as np

class Modx:
    def __init__(self, x_ul, type_mod='round'):
        self.type_mod = type_mod
        self.x_ul = x_ul
            
    def modified_x(self, x_):
        if self.type_mod == 'round':
            x_ = np.where(x_<self.x_ul[:,0], self.x_ul[:,0], x_)
            x_ = np.where(x_>self.x_ul[:,1], self.x_ul[:,1], x_)
        elif self.type_mod == 'reflect':
            x_ = np.where(x_<self.x_ul[:,0], self.x_ul[:,0] + np.abs(x_-self.x_ul[:,0]), x_)
            x_ = np.where(x_>self.x_ul[:,1], self.x_ul[:,1] - np.abs(self.x_ul[:,1]-x_), x_)
        elif self.type_mod == 'torus':
            x_ = np.where(x_<self.x_ul[:,0], self.x_ul[:,1] - np.abs(x_-self.x_ul[:,0]), x_)
            x_ = np.where(x_>self.x_ul[:,1], self.x_ul[:,0] + np.abs(self.x_ul[:,1]-x_), x_)
        elif self.type_mod == 'none':
            x_ = np.copy(x_)
        return x_