#python 3.5
#principal component analysis
import sys, os #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from reading_train_data import txs, tys
from reading_test_data import tds, tis
from configure import args_x
import numpy as np
from sklearn.decomposition import PCA



num_components = 25

pca = PCA(num_components)

x_transformed = pca.fit_transform(txs)
ts_transformed = pca.transform(tds)


from write_results import write_predictions
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_transformed, tys[:,1])

predicted = model.predict_proba(ts_transformed)

write_predictions("regression_pca25.csv",tis,predicted)
"""



writing_parts = [list(map(str,rw)) for rw in x_transformed]
labels = [ "feature" + str(i) for i in range(1,num_components+1)]
writing_parts = np.append([labels], writing_parts, axis=0)
np.savetxt("../../data/pca/training_pca" + str(num_components) + ".csv", writing_parts, fmt='%s', delimiter=',')

writing_parts = [list(map(str,rw)) for rw in ts_transformed]
labels = [ "feature" + str(i) for i in range(1,num_components+1)]
writing_parts = np.append([labels], writing_parts, axis=0)
np.savetxt("../../data/pca/test_pca" + str(num_components) + ".csv", writing_parts, fmt='%s', delimiter=',',newline='\n')
"""