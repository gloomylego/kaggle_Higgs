#Support Vector Machines
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from write_results import write_predictions, write_predictions2
from additionals import fe_rf, submit_regression

import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dxs = [np.array(txs),np.array(tds)]
dy = np.array(tys[:,1])

pca = PCA(n_components=25)
dxs[0] = pca.fit_transform(dxs[0])
dxs[1] = pca.transform(dxs[1])

"""
for i in range(2):
    
    #new_col = my_data.sum(1)[...,None]
    fcol_neg = np.array([v < 0 for v in dxs[i][:,0]])
    median = np.median(dxs[i][:,0][np.logical_not(fcol_neg)])
    assert len(dxs[i][:,0]) > 40
    dxs[i][:,0] = [median if fcol_neg[j] else dxs[i][:,0][j]
                 for j in range(len(fcol_neg))]
    fcol_neg = [[int(i)] for i in fcol_neg]
    np.append(dxs[i], fcol_neg, 1)
"""
 
dxs[0] = preprocessing.scale(dxs[0])
dxs[1] = preprocessing.scale(dxs[1])


model = LogisticRegression()
model.fit(dxs[0], dy)
predicted = model.predict_proba(dxs[1])
write_predictions("regression_.csv",tis,predicted)
