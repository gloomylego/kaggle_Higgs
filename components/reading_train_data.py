from configure import *
import numpy as np

def dictionary_y(x): 
    return 1 if x==b's' else 0

train_data_x = np.loadtxt(filename_train, delimiter=',', skiprows=1, usecols=range(len(args_x)))

train_data_y = np.loadtxt(filename_train, converters = {args_total_amount-1:dictionary_y},
                          delimiter=',', skiprows=1, usecols=range(len(args_x),args_total_amount))

txs = train_data_x[:,1:]
tys = train_data_y
#tris = list(map(int,train_data_x[:,0]))

if configure_verbose_mode:
    print("train data is loaded")