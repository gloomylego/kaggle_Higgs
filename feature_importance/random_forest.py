#python 3.5
#using http://habrahabr.ru/company/mlclass/blog/247751/
#using http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from configure import configure_verbose_mode
from reading_train_data import *
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

forest = ExtraTreesClassifier(n_estimators=500, random_state=0)
forest.fit(train_data_x[:, 1:], train_data_y[:,1]) #skip id with [:, 1:]; training only with the result value(s or b)
# display the relative importance of each attribute
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

if configure_verbose_mode:
    for i in range(len(args_x)-1):
        print("%d.  %s (%f)" % (i + 1, args_x[indices[i]+1], importances[indices[i]]))

def show_image(sorted=False):
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],0)#for yerr of plt.bar
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    if sorted:
        permuted_names = list(range(len(args_x)-1))
        for i in range(len(args_x)-1):  permuted_names [i] = (args_x[1:])[indices[i]]
        plt.bar(range(len(args_x)-1), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(args_x)-1), permuted_names, rotation=90)
    else:
        plt.bar(range(len(args_x)-1), importances,
                color="r", yerr=std, align="center")
        plt.xticks(range(len(args_x)-1), args_x[1:], rotation=90)
    plt.xlim([-1, len(args_x)-1])
    plt.show()
    return 0


show_image(True)

show_image(False)