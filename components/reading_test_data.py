from configure import *
import numpy as np

test_data = np.loadtxt(filename_test, delimiter=',', skiprows=1)
tds = test_data[:,1:]
tis = list(map(int,test_data[:,0]))

if configure_verbose_mode:
    print("test data is loaded")