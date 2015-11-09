#python 3.5
#saves(shows) histogram of every feature
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from configure import args_x
import numpy as np
import matplotlib.pyplot as plt

number_of_bars = 100

v_start = 0
v_end = len(tds[0])

i = 0
#if no path is defined then histogram will be shown

def histImage(vals, bars = number_of_bars, path = "", figure_name = ""):
    assert len(vals)>=1
    y, bins, patches = plt.hist(vals, bins = bars, color = 'red')
    plt.title("Data distribution")
    plt.suptitle(figure_name)
    if path == "":    
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


for i in range(v_start,v_end):
    stri = str((i+1)//10) + str((i+1)%10)
    histImage(txs[:,i], number_of_bars, "../calculated/features_histograms/" + stri + args_x[i+1] + ".png", args_x[i+1])