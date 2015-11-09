import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from write_results import write_predictions

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
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


write_predictions("regression.csv",tis,predicted)