#Support Vector Machines
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import train_data_x, train_data_y
from reading_test_data import test_data
from write_results import write_predictions

from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(train_data_x[:,1:], train_data_y[:,1])

predicted = model.predict_proba(test_data[:,1:])

write_predictions("cart.csv", test_data[:,0], predicted)
