#python 3.5
#using http://habrahabr.ru/company/mlclass/blog/247751/
#Recursive Feature Elimination
import numpy as np
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 1)
rfe = rfe.fit(txs, tys[:,1])
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)