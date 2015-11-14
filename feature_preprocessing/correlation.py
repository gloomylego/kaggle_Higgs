import sys, os  #is necessary for relative import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components')) #is necessary for relative import
from configure import filename_train, args_x, args_y
from numpy import savetxt
from pandas import read_csv, DataFrame, Series

dataset = read_csv(filename_train,',')#,skiprows=1,)
dataset = dataset.drop(args_x[0],1)
mapping = {'b': 0, 's': 1}

dataset[args_y[1]] = dataset[args_y[1]].apply(mapping.get)

#print(dataset.head())
corr = dataset.corr()

#map corr to string
#insert row of names
#insert col of names

savetxt("../calculated/correlation.csv",corr,'%f',',')