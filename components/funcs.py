from sklearn.linear_model import *
import numpy as np
import pandas as ps
from configure import args_x, configure_verbose_mode
from sklearn import preprocessing
from sklearn.cluster import KMeans


def k_nearest_cluster(xs,ds,n):
    model = KMeans(n_clusters=n, precompute_distances = True, n_jobs=1)#multiparallel doesn't work :(
    model.fit(xs)
    frame_x = ps.DataFrame(model.predict(xs)[None].T,columns=["Mnew_knearest" + str(n)])
    frame_x.name = "Mnew_knearest" + str(n)

    frame_d = ps.DataFrame(model.predict(ds)[None].T,columns=["Mnew_knearest" + str(n)])
    frame_d.name = "Mnew_knearest" + str(n)

    return (frame_x, frame_d)



#;2nd param: background, signal; 3,4 ~ syl for bg and signal
def get_threshold(weights,bss,cb = 0, cs = 1):
    sSelector = np.array([row == cs for row in bss])
    bSelector = np.array([row == cb for row in bss])
    ss = weights[sSelector]
    bs = weights[bSelector]
    max = np.max(ss,0)
    min = np.min(bs,0)
    return (min+max)/2#min-0.001




def regression_sgd(x, y, predict, proba = False):
    sgd = SGDRegressor(loss='huber', n_iter=100)
    sgd.fit(x, y)
    if proba:
        sgd.predict_proba(predict)
    else:
        return sgd.predict(predict)
        
def amsasimov(s,b):
        from math import sqrt,log
        if b == 0:
            return 0
        return sqrt(2*((s+b)*log(1+float(s)/b)-s))

def amsfinal(s,b):
    return amsasimov(s,b+10.)


def get_best_val(column, i):
    stri = str(i//10) + str(i%10)
    full_fname = "../calculated/features_histograms/" + stri + args_x[i] + ".csv"
    full_fname_filtered = "../calculated/features_histograms_filtered/" + stri + args_x[i] + "_f.csv"
    bars = ps.read_csv(full_fname,',',header=0).as_matrix()
    bars_filtered = ps.read_csv(full_fname_filtered,',',header=0).as_matrix()
    percentage = bars[0][2]
    i_best = -1
    i_diff = 1 #maximum as difference of possabilities
    for i in range(len(bars_filtered)):
        if abs(percentage - bars_filtered[i][2]) < i_diff:
            i_best = i
            i_diff = abs(percentage - bars_filtered[i][2])
    return (bars_filtered[i_best][0] + bars_filtered[i_best][1] / 2)


def modifycols_includecols(xs,ds,NaN):
    amount_cols = xs.shape[1]
    for i in range(amount_cols):
        if np.equal(xs[:,i],NaN).any():
            best_val_to_set = get_best_val(xs[:,i],i+1)#skip the first column(index)
            xs_col = [[j == NaN] for j in xs[:,i]]
            ds_col = [[j == NaN] for j in ds[:,i]]
            xs[:,i][xs[:,i]==NaN] = best_val_to_set
            ds[:,i][ds[:,i]==NaN] = best_val_to_set
            
            xs = np.append(xs, xs_col, 1)
            ds = np.append(ds, ds_col, 1)
    return (xs, ds)
    
    
def scaling_according_to_weights(xs,ds):
    importances = ps.read_csv("../calculated/features_importance_xgboost.csv",',',header=0)
    vals = importances[args_x[1:]].as_matrix()
    mms1 = preprocessing.MinMaxScaler()
    xs = mms1.fit_transform(xs)
    mms2 = preprocessing.MinMaxScaler()
    ds = mms2.fit_transform(ds)
    vals = np.array(list(map(float,vals[0])))
    #vals = np.array(list(map(float,vals)))
    vals /= np.max(vals)
    
    xs = np.array([np.divide(i,vals) for i in xs])
    ds = np.array([np.divide(i,vals) for i in ds])
    return (xs, ds)

def feature_importance(filename, names, booster):
    prev_fn = booster.feature_names
    
    fscore = [ (k,v) for k,v in booster.get_fscore().iteritems() ]
    fscore.sort(key=lambda x:x[1], reverse=True)
    
    
    if configure_verbose_mode:
        print(fscore)
    imps = np.array([[str(i[1]) for i in fscore]])
    imps = np.append([[i[0] for i in fscore]], imps, axis=0)
    np.savetxt(filename, imps,fmt='%s', delimiter=',')
    #return the original state
    booster.feature_names = prev_fn