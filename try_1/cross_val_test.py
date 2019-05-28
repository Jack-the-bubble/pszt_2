from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

dec_tree_1 = DecisionTreeRegressor(max_depth=7)
dec_tree_2 = DecisionTreeRegressor(max_depth=5)

x = []
y = []

for i in range(100):
	x.append(i)
	y.append(i)

x = np.array(x)
y = np.array(y)

x = x.reshape(-1, 1)

cross_val =  KFold(10, True, 1)
scores = cross_val_score(dec_tree_1, x, y, scoring='neg_mean_squared_error', cv = cross_val, n_jobs=1)

print("Folds: ", len(scores), ", mean squared error: ", np.mean(np.abs(scores)), " std: ", np.std(scores))