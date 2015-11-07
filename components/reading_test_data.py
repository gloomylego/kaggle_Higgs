from configure import *
import numpy as np

test_data = np.loadtxt(filename_test, delimiter=',', skiprows=1)

if configure_verbose_mode:
    print("test data is loaded")