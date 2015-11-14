#python 2.7
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from configure import filename_train, filename_test, args_x, args_y
from write_results import write_predictions
import numpy as np
import pandas as ps
import xgboost as xgb
from funcs import regression_sgd

get_feature_importance = False

dtrain = ps.read_csv(filename_train,',',header=0)
dtest = ps.read_csv(filename_test,',',header=0)

xs = dtrain[args_x[1:]].as_matrix()
ys = np.array([0 if i == b'b' else 1 for i in dtrain[args_y[1]]])
ds = dtest[args_x[1:]].as_matrix(columns = args_x[1:])
#def regression_sgd(x, y, predict, proba = False):
weights = dtrain[args_y[0]].as_matrix()

new_feature_train = regression_sgd(xs,weights,xs)
new_feature_test = regression_sgd(xs,weights,ds)
new_feature_train = np.array([[i] for i in new_feature_train])
new_feature_test = np.array([[i] for i in new_feature_test])

xs = np.append(xs, new_feature_train, 1)
ds = np.append(ds, new_feature_test, 1)


gbm = xgb.XGBClassifier(silent=False, nthread=4, max_depth=10, n_estimators=800, subsample=0.5, learning_rate=0.03, seed=1337)

gbm.fit(xs,ys)

if get_feature_importance:
    bst = gbm.booster()
    bst.feature_names = args_x[1:]
    imps = bst.get_fscore()
    print(imps)
else:
    predictions = gbm.predict_proba(ds)
    arr = dtest[args_x[0]].as_matrix()
    write_predictions("1boost.csv",arr,predictions)

