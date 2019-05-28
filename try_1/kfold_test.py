from sklearn.model_selection import KFold
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from data_analyzer import analyze_data
x = []
y = []

for i in range(100):
	x.append(i)
	y.append(i)

# X = np.array(x)
# Y = np.array(y)

# X = X.reshape(-1, 1)
X= analyze_data()
Y = X.iloc[:, 12]

cc = list(combinations(X.iloc[0:11], 11))
print(len(cc))

indexes = []
indexes.append(cc[0][0])
indexes.append(cc[0][0])

x_smol = X[indexes]

# print(x_smol)
cross_val =  KFold(10, True, 1)



print(cross_val)


