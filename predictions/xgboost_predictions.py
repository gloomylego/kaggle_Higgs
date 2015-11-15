#python 2.7
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from configure import filename_train, filename_test, args_x, args_y
from write_results import write_predictions
from funcs import regression_sgd, modifycols_includecols, scaling_according_to_weights
import numpy as np
import pandas as ps
import xgboost as xgb
from sklearn.cluster import KMeans


#parameters info
pdata_nearest = True
pdata_nearest_neigh = [3,7,15] #should be length of 3
pdata_nearest_scaled = True #if nearest is False, it doesn't matter
pdata_nearest_scaled_nan_to = -10#if !pdata_nearest_scaled, it doesn't matter
exclude_zd_add_cols_with_zd = False #only decreased :(
add_column_as_regression_sgd = True
pdata_sgd = 0.9#1 #percentage of used trained data for training regression_sgd

get_feature_importance = False
#const
NaN = -999
#prarameters info


dtrain = ps.read_csv(filename_train,',',header=0)
dtest = ps.read_csv(filename_test,',',header=0)

#original data
xs = dtrain[args_x[1:]].as_matrix()
ys = np.array([0 if i == b'b' else 1 for i in dtrain[args_y[1]]])
ds = dtest[args_x[1:]].as_matrix(columns = args_x[1:])
    
if pdata_nearest:
    nxs = xs
    nds = ds
    if pdata_nearest_scaled:
        nxs[nxs==NaN] = pdata_nearest_scaled_nan_to
        nds[nds==NaN] = pdata_nearest_scaled_nan_to
        (nxs, nds) = scaling_according_to_weights(nxs, nds)

    print("K nearest (",pdata_nearest_neigh[0],")")
    k1 = KMeans(n_clusters=pdata_nearest_neigh[0], precompute_distances = True, n_jobs=1)#multiparallel doesn't work :(
    k1.fit(nxs)
    print("K nearest (",pdata_nearest_neigh[1],")")
    k2 = KMeans(n_clusters=pdata_nearest_neigh[1], precompute_distances = True, n_jobs=1)
    k2.fit(nxs)
    print("K nearest (",pdata_nearest_neigh[2],")")
    k3 = KMeans(n_clusters=pdata_nearest_neigh[2], precompute_distances = True, n_jobs=1)
    k3.fit(nxs)
    ds  = np.hstack((ds,  k1.predict(nds)[None].T,  k2.predict(nds)[None].T,  k3.predict(nds)[None].T))
    xs = np.hstack((xs, k1.predict(nxs)[None].T, k2.predict(nxs)[None].T, k3.predict(nxs)[None].T))
    
if exclude_zd_add_cols_with_zd:
    (xs, ds) = modifycols_includecols(xs, ds,NaN)    

if add_column_as_regression_sgd:
    weights = dtrain[args_y[0]].as_matrix()
    aux_end = int(len(xs)*pdata_sgd)
    new_feature_train = regression_sgd(xs[:aux_end,:],weights[:aux_end],xs)
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
    #xgb.plot_importance(gbm)
    #plt.show()
