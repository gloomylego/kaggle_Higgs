import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from write_results import write_predictions,write_predictions2
from funcs import get_threshold, regression_sgd

from sklearn import metrics
from sklearn.linear_model import *
import numpy as np


predicted = regression_sgd(txs,tys[:,0],tds,False)
threshold = get_threshold(tys[:,0],tys[:,1])
write_predictions2("regression_sgd.csv",tis,predicted,threshold)

#############################
exit()
#############################
#logical EOF

model = LogisticRegression()
model.fit(txs, tys[:,1])
#print(model)
# make predictions
#expected = train_data_y[:,1]
#predicted = model.predict(test_data[:,1:])
#probability = model.decision_function(test_data[:,1:])

##with all features result is 2.01342 ~ 1508 place
#write_predictions("regression.csv",test_data[:,0],probability,predicted)
predicted = model.predict_proba(tds)

sSelector = np.array([row[1] == 1 for row in tys])
bSelector = np.array([row[1] == 0 for row in tys])

ss = tys[sSelector]
bs = tys[bSelector]
max = np.max(ss,0)[0]
min = np.min(bs,0)[0]
threshold = (min+max)/2#min-0.001



write_predictions("regression.csv",tis,predicted)