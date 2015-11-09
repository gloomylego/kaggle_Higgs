#python 3.5
#feature importance of random forest
fe_rf = [0.099039,0.073323,0.062944,0.033049,0.019231,0.015240,
0.012399,0.055468,0.031873,0.036628,0.051389,0.068414,0.021933,
0.054562,0.030284,0.026647,0.033384, 0.031799,0.026988,0.041471,
0.027156,0.029170,0.010964,0.018906,0.025929,0.019231,0.007515,
0.009262,0.007805,0.018000]
assert len(fe_rf) == 30
#feature importance of logistic regression
fe_lr_pl = [28,10,11,15,5,22,6,3,30,13,1,2,7,16,23,29,9,19,25,17,27,26,4,20,18,24,21,14,8,12]
assert len(fe_lr_pl) == 30

#regression submits. Amount of features : [4,20] 
#features were sorted by fe_rf importance from the most important to the least one
#data wasn't filtered or preprocessed
submit_regression = [[4,1.32628],[5,1.36848],[6,1.85929],[7,1.85770],[8,1.89983],[9,1.89732],
[10,2.00177],[11,2.01492],[12,2.01865],[13,2.01807],[14,2.01951],[15,1.99485],[16,1.99530],
[17,1.99575],[18,1.99591],[19,1.96067],[20,1.98939],[30,2.01342]]

#regression where there is no 13th data(counting starts from the 1(index not counted))
#features are also sorted as the previous ones
#as we can see it is a bit better
submit_regression_no13 = 2.02049