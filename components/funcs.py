from sklearn.linear_model import *
import numpy as np

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