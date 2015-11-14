import numpy as np
import pandas as ps
import matplotlib.pyplot as plt

filename = "../calculated/features_importance_xgboost.csv"
file = ps.read_csv(filename,',',header=0)

labels = file.columns.values
values = file.as_matrix()
values = values[0]
order = values.argsort()[::-1]
plt.bar(range(len(values)), values[order], color = "r")
plt.title("Feature importances with xgboost")
plt.xticks(range(len(values)), labels[order], rotation=90)
plt.show()
