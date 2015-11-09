#Support Vector Machines
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from write_results import write_predictions
from additionals import fe_rf, submit_regression

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
"""
importances = np.array(fe_rf).argsort()[::-1]
for i in range(4,21):
    
    xdat = txs[:,importances[:i]]
    ddat = tds[:,importances[:i]]
    model = LogisticRegression()
    model.fit(xdat, tys[:,1])
    predicted = model.predict_proba(ddat)
    write_predictions("../../regression" + str(i) + ".csv",tis,predicted)
"""

"""
plt.figure()
plt.title("Data submit score for regression model")
plt.suptitle("Look into additionals.py (submit_regression) for info")
s_r = np.array(submit_regression)

plt.bar(s_r[:,0], s_r[:,1], color="r", align="center")
plt.xticks(s_r[:,0], list(map(int,s_r[:,0])),rotation = 90)
plt.show()
"""

importances = np.array(fe_rf).argsort()[::-1]

importances[12] = importances[13]
i = 13
xdat = txs[:,importances[:i]]
ddat = tds[:,importances[:i]]
model = LogisticRegression()
model.fit(xdat, tys[:,1])
predicted = model.predict_proba(ddat)
write_predictions("../../regression_no_13t.csv",tis,predicted)