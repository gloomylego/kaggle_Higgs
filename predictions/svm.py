#Support Vector Machines
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from write_results import write_predictions

from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(txs, tys[:,1])

predicted = model.predict_proba(tds)

write_predictions("cart.csv", tis[:,0], predicted)
