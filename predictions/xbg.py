#python 2.7
#program needs args
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from configure import filename_train, filename_test, args_x, args_y, configure_verbose_mode, configure_release_mode
from write_results import write_predictions2,write_predictions3
from funcs import k_nearest_cluster, regression_sgd, modifycols_includecols, scaling_according_to_weights, feature_importance
from xgboost_predictions_fe import add_features
import numpy as np
import pandas as ps
import xgboost as xgb

#default args
num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
subsample = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 10
#settings
need_train = True
pdata_nearest = True
pdata_nearest_neigh = [7,15]
pdata_regression_sgd = True
pdata_additional = True
threshold = 0.14
nthread = 2

get_feature_importance = True
feature_importance_filename = "importance.csv"
model_filename = 'higgs_model'#filename to save/load
#const
NaN = -999


#reading data
dtrain = ps.read_csv(filename_train,',',header=0)
dtest = ps.read_csv(filename_test,',',header=0)
#our current set we are going to work with
#train_indexes = np.array(list(map(int,dtrain[args_x[0]])))
train_x = dtrain[args_x[1:]]

test_indices = np.array(list(map(int,dtest[args_x[0]])))
test_x = dtest[args_x[1:]]


labels = np.array([0 if i == b'b' else 1 for i in dtrain[args_y[1]]])
weights = dtrain[args_y[0]].as_matrix()

if configure_verbose_mode:
    print("Finished: reading data")


if pdata_nearest:
    nxs = train_x.as_matrix()
    nds = test_x.as_matrix()
    nxs[nxs==NaN] = -10
    nds[nds==NaN] = -10
    (nxs, nds) = scaling_according_to_weights(nxs, nds)
    
    for i in pdata_nearest_neigh:
        (kx, kd) = k_nearest_cluster(nxs, nds, i)
        train_x = train_x.join(kx)
        test_x = test_x.join(kd)
    if configure_verbose_mode:
        print("Finished: algorithm nearest")


if pdata_regression_sgd:
    nxs = dtrain[args_x[1:]].as_matrix()
    nds = dtest[args_x[1:]].as_matrix()

    weights = dtrain[args_y[0]].as_matrix()
    aux_end = int(len(nxs)*0.9)#0.9 is the best value!
    new_feature_train = regression_sgd(nxs[:aux_end,:],weights[:aux_end],nxs)
    new_feature_test = regression_sgd(nxs,weights,nds)
    new_feature_train = ps.DataFrame([i for i in new_feature_train],columns = ["Mnew_sgd"])
    new_feature_test = ps.DataFrame(([i for i in new_feature_test]),columns = ["Mnew_sgd"])
    train_x = train_x.join(new_feature_train)
    test_x = test_x.join(new_feature_test)
    if configure_verbose_mode:
        print("Finished: regression sgd")

if pdata_additional:
    train_x = add_features(train_x)
    test_x = add_features(test_x)
    if configure_verbose_mode:
        print("Finished: feature engeneering")

#if exclude_sth:
exclusions = ['PRI_met_phi', 'PRI_lep_phi', 'PRI_tau_phi', 'PRI_jet_leading_phi','PRI_jet_subleading_phi',
              'PRI_tau_eta','PRI_lep_eta',
              'PRI_jet_subleading_pt','PRI_jet_subleading_p_tot','PRI_jet_subleading_px','PRI_jet_subleading_eta',
              'PRI_jet_all_pt','PRI_jet_subleading_py','New_jet_delta_eta','New_sum_jet_pt','PRI_jet_subleading_pz',
              'New_ht','PRI_jet_num','New_ht_met','New_lep_met_deltaphi',
              #'New_frac_lep_pt','New_frac_lep_p' #2 and 1 valued by 3000 nr tree
              ]
cols_names = [c for c in train_x.columns.values if c not in exclusions]
train_x = train_x[cols_names]
test_x = test_x[cols_names]
if need_train:
    sum_w_signal = sum( weights[i] for i in range(len(labels)) if labels[i] == 1.0  )
    sum_w_background = sum( weights[i] for i in range(len(labels)) if labels[i] == 0.0  )

    param = {}
    param['objective'] = 'binary:logitraw'
    param['scale_pos_weight'] = sum_w_background/sum_w_signal
    param['bst:eta'] = eta
    param['bst:max_depth'] = max_depth
    param['bst:subsample'] = subsample
    param['eval_metric'] = 'ams@' + str(threshold)
    param['silent'] = 1
    param['nthread'] = nthread
    plst = list(param.items())#+[('eval_metric', 'ams@0.15')]

    dmat = xgb.DMatrix(train_x, label=labels, weight=weights, missing = np.nan)
    watchlist = [(dmat,'train')]
    bst = xgb.train(plst, dmat, num_rounds, watchlist)
    bst.save_model(model_filename)
    #bst.save_model('higgs.model.%dstep.depth%s'%(num_round,max_depth))
    if configure_verbose_mode:
        print("model %s saved" % (model_filename))
    if get_feature_importance:
        feature_importance(feature_importance_filename, train_x.columns.values, bst)
    

bst = xgb.Booster({'nthread':nthread})
bst.load_model(model_filename)

predicted = bst.predict(xgb.DMatrix(test_x,missing = np.nan))

write_predictions3("boost.csv",test_indices,predicted, threshold)