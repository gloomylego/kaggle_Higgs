import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from write_results import write_predictions, write_predictions2
import numpy as np

from sklearn import metrics
from sklearn import linear_model

# sooo baaad

model = linear_model.BayesianRidge()
#model = linear_model.LassoLars()
#model = linear_model.RidgeCV()
#model = linear_model.Ridge (alpha = .5)
#model = linear_model.LinearRegression()
model.fit(txs, tys[:,0])

sSelector = np.array([row[1] == 1 for row in tys])
bSelector = np.array([row[1] == 0 for row in tys])

ss = tys[sSelector]
bs = tys[bSelector]
max = np.max(ss,0)[0]
min = np.min(bs,0)[0]
threshold = (min+max)/2#min-0.001

predicted = model.predict(tds)

write_predictions2("least_squares_reg.csv", tis, predicted, threshold)
