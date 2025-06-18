#coding: utf-8
import pandas as pd
import numpy as np
import gc
import os
from os import path

class import_data_class:
    def read_file(self, filepath):
        df = pd.read_csv(filepath)
        clm_name = df.columns.values
        df = df.drop("Unnamed: 0", axis=1)
        data = df.values    
        del df
        gc.collect()
        return data
    
    def get_dataset(self, run, type_):
        if type_ == 1:
            work_path = path.dirname( path.abspath(__file__) )
            os.chdir(work_path)
            dir_base = os.path.abspath('..\\..\\..\\..\\..\\') + '\\data\\binary_data\\'
            filename = "binary_dataset" + str(run) + ".csv"
            data = self.read_file(dir_base + filename)
        elif type_ == 2:
            data = np.random.rand(1000, 1000)
        return data