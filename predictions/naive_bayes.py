import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import train_data_x, train_data_y
from reading_test_data import test_data

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(train_data_x[:,1:], train_data_y[:,1])
print(model)
# make predictions
expected = train_data_y[:,1]
predicted = model.predict(train_data_x[:,1:])
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))