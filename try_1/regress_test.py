from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import pyplot as plt

x = []
y = []
for i in range(20):
	x.append(i)
	y.append(i)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
X = np.array(x)
Y = np.array(y)
X = X.reshape(-1,1)

# x.reshape(-1, 1)
regr_1.fit(X, Y)
regr_2.fit(X, Y)

X_test = np.arange(0, 5, 0.01)[:, np.newaxis]
y1 = regr_1.predict(X_test)
y2 = regr_2.predict(X_test)


plt.figure()
print("y1: ", y1, "\ny2: ", y2)
plt.plot(X_test, y1)
plt.plot(X_test, y2)
plt.show()

