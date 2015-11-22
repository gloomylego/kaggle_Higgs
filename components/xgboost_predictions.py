#python 2.7
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from configure import filename_train, filename_test, args_x, args_y, configure_verbose_mode, configure_release_mode
from write_results import write_predictions
from funcs import k_nearest_cluster, regression_sgd, modifycols_includecols, scaling_according_to_weights
from xgboost_predictions_fe import add_features
import numpy as np
import pandas as ps
import xgboost as xgb


#parameters info
pdata_nearest = True
pdata_nearest_neigh = [7,15] #3 nearest has the most awful importance
pdata_nearest_scaled = True #if nearest is False, it doesn't matter
pdata_nearest_scaled_nan_to = -10#if !pdata_nearest_scaled, it doesn't matter
exclude_zd_add_cols_with_zd = False #only decreased :(
pdata_add_features = True
add_column_as_regression_sgd = True
pdata_sgd = 0.9#1 #percentage of used trained data for training regression_sgd
#threshold_filter_features = 0.05 #used to filter features, that are less then this value
get_feature_importance = False   #rewrite file of feature importances
feature_importance_file = "../calculated/fe_xgb.csv" if configure_release_mode else "fe_xgb.csv"
#const
NaN = -999
#prarameters info


dtrain = ps.read_csv(filename_train,',',header=0)
dtest = ps.read_csv(filename_test,',',header=0)
###
###our current set we are going to work with
train_indexes = dtrain[args_x[0]]
train_x = dtrain[args_x[1:]]

test_indexes = dtest[args_x[0]]
test_x = dtest[args_x[1:]]

labels = np.array([0 if i == b'b' else 1 for i in dtrain[args_y[1]]])
weights = dtrain[args_y[0]].as_matrix()

#original data
#xs = dtrain[args_x[1:]].as_matrix()
#ys = np.array([0 if i == b'b' else 1 for i in dtrain[args_y[1]]])
#ds = dtest[args_x[1:]].as_matrix(columns = args_x[1:])

if pdata_nearest:
    nxs = train_x.as_matrix()
    nds = test_x.as_matrix()
    if pdata_nearest_scaled:
        nxs[nxs==NaN] = pdata_nearest_scaled_nan_to
        nds[nds==NaN] = pdata_nearest_scaled_nan_to
        (nxs, nds) = scaling_according_to_weights(nxs, nds)
    
    for i in pdata_nearest_neigh:
        if configure_verbose_mode:
            print("K nearest (",i,")")
        (kx, kd) = k_nearest_cluster(nxs, nds, i)
        train_x = train_x.join(kx)
        test_x = test_x.join(kd)
    
###doesn't work now. Shouldn't be run
if exclude_zd_add_cols_with_zd:
    assert 1 == 2
    nxs = dtrain[args_x[1:]].as_matrix()
    nds = dtest[args_x[1:]].as_matrix()
    (nxs, nds) = modifycols_includecols(nxs, nds, NaN)    

###
if pdata_add_features:
    train_x = add_features(train_x)
    test_x = add_features(test_x)

if add_column_as_regression_sgd:
    #with all data
    #nxs = train_x.as_matrix()
    #nds = test_x.as_matrix()
    #or not
    nxs = dtrain[args_x[1:]].as_matrix()
    nds = dtest[args_x[1:]].as_matrix()

    weights = dtrain[args_y[0]].as_matrix()
    aux_end = int(len(nxs)*pdata_sgd)
    new_feature_train = regression_sgd(nxs[:aux_end,:],weights[:aux_end],nxs)
    new_feature_test = regression_sgd(nxs,weights,nds)
    new_feature_train = ps.DataFrame([i for i in new_feature_train],columns = ["Mnew_sgd"])
    new_feature_test = ps.DataFrame(([i for i in new_feature_test]),columns = ["Mnew_sgd"])
    train_x = train_x.join(new_feature_train)
    test_x = test_x.join(new_feature_test)
    #xs = np.append(xs, new_feature_train, 1)
    #ds = np.append(ds, new_feature_test, 1)

#Jet_num
#New_sum_jet_pt
#new_frac_lep_pt
#if threshold_filter_features>0:
#    fs = ps.read_csv(feature_importance_file,',',header=0)
#    print(len(fs.columns.values))
#    fs = fs.dropna(axis=1,how='all')
#    print(len(fs.columns.values))
#    vals = np.array(list(map(float,fs.as_matrix()[0])))
#    vals /= vals.max()
    
#    for (iter,i) in zip(range(len(vals)),fs.columns.values):
#        if i in train_x and vals[iter] < threshold_filter_features:
#            train_x = train_x.drop(i,1)
#            test_x = test_x.drop(i,1)

#excludes = [
#    'New_jet_delta_eta', 'New_lep_met_deltaphi' #below 100  
#    ,'PRI_jet_num', 'PRI_jet_subleading_pz', 'New_sum_jet_pt', 'New_ht', 'New_ht_met', 'New_lep_met_deltaphi' #below 400
#    ,'PRI_met_phi', 'PRI_lep_phi', 'PRI_tau_phi', 'PRI_jet_leading_phi','PRI_jet_subleading_phi' #add features
#    ,'PRI_tau_eta','PRI_lep_eta'
#    ]
excludes = ['PRI_met_phi', 'PRI_lep_phi', 'PRI_tau_phi', 'PRI_jet_leading_phi','PRI_jet_subleading_phi',
              'PRI_tau_eta','PRI_lep_eta'] 
cols_names = [c for c in train_x.columns.values if c not in excludes]
train_x = train_x[cols_names]
test_x = test_x[cols_names]

#sum_w_signal = sum( weights[i] for i in range(len(labels)) if labels[i] == 1.0  )
#sum_w_background = sum( weights[i] for i in range(len(labels)) if labels[i] == 0.0  )



gbm = xgb.XGBClassifier(silent=not configure_verbose_mode, nthread=4, max_depth=10, n_estimators=3000, subsample=0.9,
                        learning_rate=0.01,seed=1337)#,objective='binary:logitraw')#, scale_pos_weight = sum_w_background/sum_w_signal)
if configure_verbose_mode:
    print gbm

gbm.fit(train_x,labels,weights)


#watchlist = [ (xgmat,'train') ]
##bst = xgb.train( plst, xgmat, num_round, watchlist );
#matrix = xgb.DMatrix(train_x,labels,weight = weights)


#100 features maximum :(
if get_feature_importance:
    prev_fn = bst.feature_names
    bst = gbm.booster()
    bst.feature_names = [i for i in train_x.columns.values]#args_x[1:]
    
    imps = bst.get_fscore()
    #print (len(imps))
    if configure_verbose_mode:
        print(imps)
    imps = np.array([[str(imps[i]) if imps.has_key(i) else "NaN" for i in bst.feature_names]])
    imps = np.append([train_x.columns.values], imps, axis=0)
    np.savetxt(feature_importance_file, imps,fmt='%s', delimiter=',')
    #return the original state
    bst.feature_names = prev_fn


predictions = gbm.predict_proba(test_x)
arr = dtest[args_x[0]].as_matrix()
write_predictions("1boost.csv",arr,predictions)

#import matplotlib.pyplot as plt
#xgb.plot_importance(gbm)
#plt.show()
#bad plot :(